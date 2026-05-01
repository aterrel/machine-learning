# Gaussian Naive Bayes: Algorithm

Gaussian Naive Bayes is a generative classifier. It assumes each feature is conditionally independent given the class (the "naive" assumption) and that the class-conditional distribution of each feature is Gaussian. Training computes per-class feature means, variances, and log-prior probabilities from the training set. Inference evaluates the log-probability of each test sample under each class Gaussian and returns the argmax class.

The inference step — computing `(n_test × n_classes)` log-probabilities — is embarrassingly parallel. Every `(sample, class)` pair is independent, making it ideal for GPU acceleration. Training is a single pass over the training set and runs on the CPU.

```
Training (CPU):
  for each class c:
    mask = (y_train == c)
    means[c]  = X_train[mask].mean(axis=0)
    vars[c]   = X_train[mask].var(axis=0) + 1e-9  # epsilon for stability
    log_prior[c] = log(count[c] / n_train)

Inference (GPU):
  for each (sample i, class c):
    log_prob[i,c] = log_prior[c]
                  + sum_f( -0.5*log(2π*var[c,f])
                           - 0.5*(x[i,f] - mean[c,f])^2 / var[c,f] )
  prediction[i] = argmax_c log_prob[i,c]
```

**Source:** `demos/05_naive_bayes/gpu_nb.py:66–100`
