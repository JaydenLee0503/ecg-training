"""Circuit derivatives, persistence, optimizer scheduling, and patient boundaries."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ecgvmd.reupload import compact_qnode, ReuploadVQC
from scripts.improve_vqc import choose_candidate, nested_splits


class ReuploadTests(unittest.TestCase):
    def test_batched_backends_and_finite_difference_gradient(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(3,4))*.5
        w = pnp.array(rng.normal(size=(2,2,3))*.2,requires_grad=True)
        native = compact_qnode(2,2,True)
        backprop = compact_qnode(2,2,True,device='default.qubit',diff_method='backprop')
        np.testing.assert_allclose(np.asarray(native(X,w)),np.asarray(backprop(X,w)),atol=1e-12,rtol=0)
        singles = np.asarray([native(x,w) for x in X]).T
        np.testing.assert_allclose(np.asarray(native(X,w)),singles,atol=1e-12,rtol=0)
        def objective(weights):
            z = pnp.stack(backprop(X,weights))
            return pnp.sum(z**2)/z.size
        gradient = np.asarray(qml.grad(objective)(w))
        numerical = np.empty(w.shape)
        for index in np.ndindex(w.shape):
            a,b = np.array(w),np.array(w)
            a[index] += 1e-6
            b[index] -= 1e-6
            numerical[index] = (float(objective(a))-float(objective(b)))/2e-6
        np.testing.assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-5)

    def test_fit_checkpoint_reload_and_validation_does_not_train(self):
        X = np.random.default_rng(5).normal(size=(12,4))*.5
        y = np.array(['ARR','CHF','NSR']*4)
        model = ReuploadVQC(n_qubits=2,epochs=3,batch_size=6)
        model.fit(X,y,eval_set=(X[:3],y[:3]),eval_epochs=(1,3))
        without = ReuploadVQC(n_qubits=2,epochs=3,batch_size=6).fit(X,y)
        np.testing.assert_array_equal(model.w_,without.w_)
        np.testing.assert_array_equal(model.W_,without.W_)
        self.assertEqual(model.n_steps_,6)
        self.assertEqual(set(model.validation_predictions_),{1,3})
        self.assertTrue(np.isfinite([r['gradient_norm'] for r in model.history_]).all())
        self.assertEqual(len(model.loss_),3)
        np.testing.assert_allclose(model.predict_proba(X).sum(1),1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            joblib.dump(model,path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(model.decision_function(X),loaded.decision_function(X),atol=1e-12)
        model.w_,model.W_,model.b_ = model.validation_weights_[1]
        np.testing.assert_array_equal(model.predict(X[:3]),model.validation_predictions_[1])

    def test_optimizer_schedule_prefix_and_backend_fit_equivalence(self):
        X = np.random.default_rng(2).normal(size=(12,4))*.4
        y = np.array(['ARR','CHF','NSR']*4)
        short = ReuploadVQC(n_qubits=2,epochs=1,batch_size=6).fit(X,y)
        longer = ReuploadVQC(n_qubits=2,epochs=2,batch_size=6).fit(
            X,y,eval_set=(X,y),eval_epochs=(1,))
        np.testing.assert_array_equal(short.w_,longer.validation_weights_[1][0])
        adjoint = ReuploadVQC(n_qubits=2,epochs=1,batch_size=6,device='lightning.qubit',diff_method='adjoint').fit(X,y)
        np.testing.assert_allclose(short.w_,adjoint.w_,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(short.W_,adjoint.W_,atol=1e-9,rtol=1e-9)

    def test_nested_patients_are_disjoint_and_covered(self):
        y = np.repeat(['ARR','CHF','NSR'],60)
        patients = np.concatenate([np.repeat([f'{c}/{i}' for i in range(15)],4) for c in ('ARR','CHF','NSR')])
        outer = np.tile(np.repeat(np.arange(5),12),3)
        for development,test,inner in nested_splits(y,patients,outer).values():
            seen=[]
            for train,val in inner:
                self.assertFalse(set(patients[train]) & set(patients[val]))
                self.assertFalse(set(patients[train]) & set(patients[test]))
                self.assertFalse(set(patients[val]) & set(patients[test]))
                seen.extend(val)
            np.testing.assert_array_equal(np.sort(seen),development)

    def test_selection_uses_declared_score_and_tie_break(self):
        candidates=[dict(macro_f1=.7,n_layers=3,epochs=20,candidate_order=2),
                    dict(macro_f1=.7,n_layers=2,epochs=40,candidate_order=0),
                    dict(macro_f1=.7,n_layers=2,epochs=20,candidate_order=1)]
        self.assertEqual(choose_candidate(candidates),candidates[2])
        candidates[0]['macro_f1']=.71
        self.assertEqual(choose_candidate(candidates),candidates[0])


if __name__=='__main__':
    unittest.main()
