"""Compact VQC with repeated data encoding and a declared observable readout.

This experimental classifier leaves the original VQC and saved models unchanged.
It expects already-selected, angle-scaled features. Selection/scaling belong inside
each training fold. No hardware-speed or quantum-advantage claim is made.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted


def compact_qnode(n_qubits, n_layers, reupload=True, device='lightning.qubit',
                  diff_method='adjoint'):
    import pennylane as qml

    dev = qml.device(device, wires=n_qubits, shots=None)
    pairs = sorted({tuple(sorted((i, (i+1) % n_qubits))) for i in range(n_qubits)})

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x, weights):
        for layer in range(n_layers):
            if reupload or layer == 0:
                for wire in range(n_qubits):
                    qml.RY(x[..., wire], wires=wire)
                    qml.RZ(x[..., wire+n_qubits], wires=wire)
            for wire in range(n_qubits):
                qml.Rot(*weights[layer, wire], wires=wire)
            for wire in range(n_qubits):
                qml.CNOT(wires=[wire, (wire+1) % n_qubits])
        return ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i,j in pairs])

    return circuit


class ReuploadVQC(ClassifierMixin, BaseEstimator):
    def __init__(self, n_qubits=6, n_layers=2, reupload=True, epochs=80,
                 schedule_epochs=80, batch_size=32, lr=0.02, min_lr=0.002,
                 seed=0, class_weight='balanced', verbose=False,
                 device='default.qubit', diff_method='backprop'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reupload = reupload
        self.epochs = epochs
        self.schedule_epochs = schedule_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.seed = seed
        self.class_weight = class_weight
        self.verbose = verbose
        self.device = device
        self.diff_method = diff_method

    def learning_rate(self, epoch):
        position = min(epoch/max(self.schedule_epochs-1, 1), 1.0)
        return self.min_lr + .5*(self.lr-self.min_lr)*(1+np.cos(np.pi*position))

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2*self.n_qubits:
            raise ValueError(f'Expected a matrix with {2*self.n_qubits} angle features')
        if not np.isfinite(X).all():
            raise ValueError('Non-finite angle input')
        return X

    def fit(self, X, y, eval_set=None, eval_epochs=()):
        """Fit; optional eval_set is strictly an inner-validation diagnostic.

        Validation never affects gradient updates or stops training. The experiment
        runner uses the saved checkpoint predictions for nested epoch selection.
        Outer fits supply no eval_set.
        """
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
        evaluation = None
        if eval_set is not None:
            evaluation = (self._validate_X(eval_set[0]), np.asarray(eval_set[1]))
            if len(evaluation[0]) != len(evaluation[1]):
                raise ValueError('Validation labels/features differ in length')
        eval_epochs = set(eval_epochs)
        optimizer = qml.AdamOptimizer(self.lr)
        started = time.monotonic()

        for epoch in range(self.epochs):
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
                       seconds=time.monotonic()-started)
            self.loss_.append(row['loss'])
            if evaluation is not None and epoch+1 in eval_epochs:
                pred = self.predict(evaluation[0])
                self.validation_predictions_[epoch+1] = pred
                self.validation_weights_[epoch+1] = tuple(np.asarray(v).copy() for v in (weights,head,bias))
                metrics = dict(accuracy=accuracy_score(evaluation[1],pred),
                               macro_f1=f1_score(evaluation[1],pred,average='macro',labels=self.classes_))
                self.validation_metrics_[epoch+1] = metrics
                row.update({f'val_{k}':float(v) for k,v in metrics.items()})
            self.history_.append(row)
            if self.verbose and ((epoch+1)%20==0 or epoch==0 or epoch+1==self.epochs):
                print(row,flush=True)
        return self

    def _qnode(self):
        if getattr(self,'_circuit',None) is None:
            self._circuit = compact_qnode(self.n_qubits,self.n_layers,self.reupload,
                                         self.device,self.diff_method)
        return self._circuit

    def decision_function(self, X):
        check_is_fitted(self,'w_')
        X = self._validate_X(X)
        from pennylane import numpy as pnp
        circuit = self._qnode()
        output = []
        for start in range(0,len(X),256):
            angles = pnp.array(X[start:start+256],requires_grad=False)
            z = np.asarray(pnp.stack(circuit(angles,self.w_)).T)
            output.append(z @ np.asarray(self.W_)+np.asarray(self.b_))
        return np.vstack(output) if output else np.empty((0,len(self.classes_)))

    def predict_proba(self, X):
        logits = self.decision_function(X)
        logits -= logits.max(axis=1,keepdims=True)
        probability = np.exp(logits)
        return probability/probability.sum(axis=1,keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X),axis=1)]

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k!='_circuit'}

    def save_weights(self, path):
        check_is_fitted(self,'w_')
        np.savez_compressed(path,w=np.asarray(self.w_),W=np.asarray(self.W_),b=np.asarray(self.b_),
                            classes=self.classes_,class_weight=self.class_weight_,
                            n_qubits=self.n_qubits,n_layers=self.n_layers,reupload=self.reupload,
                            epochs=self.epochs,schedule_epochs=self.schedule_epochs,
                            lr=self.lr,min_lr=self.min_lr,batch_size=self.batch_size,seed=self.seed)
