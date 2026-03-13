(defpackage #:cl-acorn/tests/conditions-test
  (:use #:cl #:rove #:cl-acorn/tests/util))

(in-package #:cl-acorn/tests/conditions-test)

;;; A log-pdf-fn that always returns +infinity (triggers initial-params error)
(defun always-bad-log-pdf (params)
  (declare (ignore params))
  (/ 1.0d0 0.0d0))

;;; A log-pdf-fn that is bad for large params but OK near zero
(defun conditional-log-pdf (params)
  (let ((p (first params)))
    (if (> (abs p) 100.0d0)
        (/ 1.0d0 0.0d0)
        (ad:* -0.5d0 (ad:* p p)))))

(deftest test-acorn-error-hierarchy
  (testing "condition hierarchy is correct"
    (ok (subtypep 'infer:model-error 'infer:acorn-error))
    (ok (subtypep 'infer:inference-error 'infer:acorn-error))
    (ok (subtypep 'infer:invalid-initial-params-error 'infer:inference-error))
    (ok (subtypep 'infer:invalid-parameter-error 'infer:model-error))
    (ok (subtypep 'infer:log-pdf-domain-error 'infer:model-error))
    (ok (subtypep 'infer:high-divergence-warning 'warning))))

(deftest test-invalid-initial-params-error-signaled-hmc
  (testing "hmc signals invalid-initial-params-error for non-finite initial log-pdf"
    (let ((errored nil))
      (handler-case
          (infer:hmc #'always-bad-log-pdf '(0.0d0)
                     :n-samples 10 :n-warmup 5)
        (infer:invalid-initial-params-error ()
          (setf errored t)))
      (ok errored))))

(deftest test-invalid-initial-params-error-signaled-nuts
  (testing "nuts signals invalid-initial-params-error for non-finite initial log-pdf"
    (let ((errored nil))
      (handler-case
          (infer:nuts #'always-bad-log-pdf '(0.0d0)
                      :n-samples 10 :n-warmup 5)
        (infer:invalid-initial-params-error ()
          (setf errored t)))
      (ok errored))))

(deftest test-use-fallback-params-restart
  (testing "use-fallback-params restart recovers and returns valid samples"
    (multiple-value-bind (samples ar diag)
        (handler-bind ((infer:invalid-initial-params-error
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:use-fallback-params '(0.0d0)))))
          (infer:hmc #'conditional-log-pdf '(999.0d0)
                     :n-samples 100 :n-warmup 50))
      (ok (> (length samples) 0))
      (ok (> ar 0.0d0))
      (ok (infer:inference-diagnostics-p diag)))))

(deftest test-return-empty-samples-restart
  (testing "return-empty-samples restart returns empty list without error"
    (multiple-value-bind (samples ar diag)
        (handler-bind ((infer:invalid-initial-params-error
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:return-empty-samples))))
          (infer:hmc #'always-bad-log-pdf '(0.0d0)
                     :n-samples 10 :n-warmup 5))
      (ok (null samples))
      (ok (= ar 0.0d0))
      (ok (infer:inference-diagnostics-p diag)))))

(deftest test-high-divergence-warning-signaled
  (testing "nuts signals high-divergence-warning for absurdly large step-size"
    (let ((warned nil))
      (handler-bind ((infer:high-divergence-warning
                       (lambda (c)
                         (declare (ignore c))
                         (setf warned t)
                         (invoke-restart 'infer:continue-with-warnings))))
        (infer:nuts (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))
                    '(0.0d0)
                    :n-samples 50 :n-warmup 5
                    :step-size 100.0d0 :adapt-step-size nil))
      (ok warned))))

(deftest test-continue-with-warnings-restart
  (testing "continue-with-warnings restart suppresses high-divergence-warning"
    (ok (handler-bind ((infer:high-divergence-warning
                         (lambda (c)
                           (declare (ignore c))
                           (invoke-restart 'infer:continue-with-warnings))))
          (infer:nuts (lambda (p) (ad:* -0.5d0 (ad:* (first p) (first p))))
                      '(0.0d0)
                      :n-samples 50 :n-warmup 5
                      :step-size 100.0d0 :adapt-step-size nil)
          t))))
