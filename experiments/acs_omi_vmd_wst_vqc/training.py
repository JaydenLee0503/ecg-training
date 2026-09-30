"""ACS-local epoch recovery for the unchanged compact VQC training algorithm.

The fit loop is copied from ecgvmd.reupload.ReuploadVQC with checkpoint I/O added.
Parity and interrupted-resume tests guard the numerical behavior. Validation data
never enters this loop. The shared implementation remains frozen.
"""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from ecgvmd.reupload import ReuploadVQC, compact_qnode
from .scripts.experiment import file_hash, atomic_json


def input_hash(X,y):
    digest=hashlib.sha256()
    for values in (X,y):
        values=np.ascontiguousarray(values)
        digest.update(str((values.shape,values.dtype.str)).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_joblib(path,value):
    temporary=path.with_suffix('.tmp')
    joblib.dump(value,temporary,compress=3)
    temporary.replace(path)


class CheckpointVQC(ReuploadVQC):
    def fit(self, X, y, checkpoint_dir, context_sha, stop_after=None):
        """Fit the frozen algorithm with atomic epoch checkpoints and exact resume."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        import pennylane as qml
        from pennylane import numpy as pnp

        if self.n_qubits < 2 or self.n_layers < 1 or self.epochs < 1:
            raise ValueError('Need at least two qubits, one layer, and one epoch')
        if self.batch_size < 1 or not 0 < self.min_lr <= self.lr:
            raise ValueError('Invalid batch size or learning rate')
        X = self._validate_X(X)
        y = np.asarray(y)
        if len(y) != len(X) or len(y) == 0:
            raise ValueError('Labels and features must have the same nonzero length')
        self.classes_, yi = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Training requires at least two classes')
        self.n_features_in_ = X.shape[1]
        self.n_qubits_ = self.n_qubits
        if self.class_weight == 'balanced':
            cw = len(y)/(len(self.classes_)*np.bincount(yi))
        elif self.class_weight is None:
            cw = np.ones(len(self.classes_))
        else:
            raise ValueError("class_weight must be 'balanced' or None")
        self.class_weight_ = cw
        sample_weights = cw[yi]
        Y = np.eye(len(self.classes_))[yi]
        self._circuit = compact_qnode(self.n_qubits, self.n_layers, self.reupload,
                                     self.device, self.diff_method)
        rng = np.random.default_rng(self.seed)
        shuffle_rng = np.random.default_rng(self.seed+10000)
        n_pairs = self.n_qubits if self.n_qubits > 2 else 1
        self.n_observables_ = 3*self.n_qubits+n_pairs
        weights = pnp.array(rng.normal(0,.1,(self.n_layers,self.n_qubits,3)),requires_grad=True)
        head = pnp.array(rng.normal(0,.1,(self.n_observables_,len(self.classes_))),requires_grad=True)
        bias = pnp.array(np.zeros(len(self.classes_)),requires_grad=True)
        self.parameter_count_ = int(weights.size+head.size+bias.size)
        self.history_, self.loss_, self.validation_predictions_ = [], [], {}
        self.validation_metrics_ = {}
        self.validation_weights_ = {}
        self.n_steps_ = 0
        optimizer = qml.AdamOptimizer(self.lr)
        completed_epochs = 0
        previous_seconds = 0.
        markers = sorted(checkpoint_dir.glob('epoch_*.json'))
        if markers:
            marker = json.loads(markers[-1].read_text())
            path = checkpoint_dir/marker['file']
            if marker['context_sha'] != context_sha or file_hash(path) != marker['sha256']:
                raise ValueError('Changed checkpoint context or bytes')
            state = joblib.load(path)
            if state['settings'] != self.get_params() or state['input_sha'] != input_hash(X,y):
                raise ValueError('Changed classifier settings or training inputs')
            weights, head, bias = (pnp.array(v,requires_grad=True) for v in state['parameters'])
            optimizer = state['optimizer']
            shuffle_rng.bit_generator.state = state['shuffle_state']
            self.history_ = state['history']
            self.loss_ = [row['loss'] for row in self.history_]
            self.n_steps_ = state['steps']
            completed_epochs = state['epoch']
            previous_seconds = self.history_[-1]['seconds']
            self.w_, self.W_, self.b_ = weights, head, bias
        started = time.monotonic()

        for epoch in range(completed_epochs,self.epochs):
            optimizer.stepsize = self.learning_rate(epoch)
            order = shuffle_rng.permutation(len(X))
            total_loss, norm_sum = 0., 0.
            count = 0
            for start in range(0,len(X),self.batch_size):
                idx = order[start:start+self.batch_size]
                xb = pnp.array(X[idx],requires_grad=False)
                yb = pnp.array(Y[idx],requires_grad=False)
                sw = pnp.array(sample_weights[idx],requires_grad=False)

                def objective(w,h,b):
                    observations = pnp.stack(self._circuit(xb,w)).T
                    logits = observations @ h+b
                    centered = logits-pnp.max(logits,axis=1,keepdims=True)
                    logp = centered-pnp.log(pnp.sum(pnp.exp(centered),axis=1,keepdims=True))
                    return -pnp.sum(sw*pnp.sum(yb*logp,axis=1))/pnp.sum(sw)

                # Explicitly retain the gradients for diagnostics; Adam applies the
                # same derivatives without a second circuit/gradient evaluation.
                gradients, loss = optimizer.compute_grad(objective,(weights,head,bias),{})
                grad_norm = float(np.sqrt(sum(np.sum(np.asarray(g)**2) for g in gradients)))
                if not np.isfinite(float(loss)) or not np.isfinite(grad_norm):
                    raise FloatingPointError(f'Non-finite training state at epoch {epoch+1}')
                weights, head, bias = optimizer.apply_grad(gradients,(weights,head,bias))
                total_loss += float(loss)*len(idx)
                norm_sum += grad_norm
                count += 1
                self.n_steps_ += 1
            self.w_, self.W_, self.b_ = weights, head, bias
            row = dict(epoch=epoch+1,steps=self.n_steps_,lr=float(optimizer.stepsize),
                       loss=total_loss/len(X),gradient_norm=norm_sum/count,
                       seconds=previous_seconds+time.monotonic()-started)
            self.loss_.append(row['loss'])
            self.history_.append(row)
            filename = f'epoch_{epoch+1:03d}.joblib'
            state = dict(epoch=epoch+1, settings=self.get_params(), input_sha=input_hash(X,y),
                         parameters=tuple(np.asarray(v) for v in (weights,head,bias)),
                         optimizer=optimizer, shuffle_state=shuffle_rng.bit_generator.state,
                         history=self.history_, steps=self.n_steps_)
            atomic_joblib(checkpoint_dir/filename,state)
            atomic_json(checkpoint_dir/f'epoch_{epoch+1:03d}.json',dict(
                file=filename,sha256=file_hash(checkpoint_dir/filename),context_sha=context_sha))
            print(f'epoch {epoch+1}/{self.epochs}: loss={row["loss"]:.6f}, seconds={row["seconds"]:.1f}',flush=True)
            if stop_after is not None and epoch+1 >= stop_after:
                break
            if self.verbose and ((epoch+1)%20==0 or epoch==0 or epoch+1==self.epochs):
                print(row,flush=True)
        return self

