"""Compare OLS, PCR and a hybrid alternating estimator.

Usage:
  python -m src.experiments.compare_methods --mode synthetic
  python -m src.experiments.compare_methods --mode nba --csv data/nba/nba_draft_combine_all_years.csv

The script implements:
 - generate synthetic covariance with controlled eigenspectrum
 - sample data with beta aligned to small or large eigen-directions
 - fit OLS, PCR (fixed r), and a hybrid alternating estimator
 - report test MSEs and optionally save a small plot
"""
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv
import math
from pathlib import Path


def generate_X_sigma(p=10, eigvals=None, random_state=0):
    rng = np.random.RandomState(random_state)
    if eigvals is None:
        # default: rapid decay, one small direction
        eigvals = np.concatenate([np.linspace(5, 1, p - 1), [0.1]])
    Q = np.linalg.qr(rng.randn(p, p))[0]
    Sigma = Q @ np.diag(eigvals) @ Q.T
    return Sigma


def sample_data(n, p, Sigma, beta_dir='small', snr=5.0, random_state=0):
    rng = np.random.RandomState(random_state)
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    evals, evecs = np.linalg.eigh(Sigma)
    if beta_dir == 'small':
        beta_vec = evecs[:, np.argmin(evals)]
    elif beta_dir == 'large':
        beta_vec = evecs[:, np.argmax(evals)]
    elif beta_dir == 'random':
        beta_vec = rng.randn(p)
        beta_vec /= np.linalg.norm(beta_vec)
    else:
        raise ValueError('beta_dir must be small/large/random')
    # scale beta amplitude to obtain desired SNR: var(X beta) = beta^T Sigma beta
    signal_var = beta_vec.T @ Sigma @ beta_vec
    sigma_eps = np.sqrt(signal_var / snr)
    beta = beta_vec.copy()  # unit-length direction
    eps = rng.normal(scale=sigma_eps, size=n)
    y = X @ beta + eps
    return X, y, beta, sigma_eps


def fit_ols(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred), model


def fit_pcr(X_train, y_train, X_test, y_test, r):
    pca = PCA(n_components=r).fit(X_train)
    Z_train = pca.transform(X_train)
    Z_test = pca.transform(X_test)
    model = LinearRegression().fit(Z_train, y_train)
    y_pred = model.predict(Z_test)
    return mean_squared_error(y_test, y_pred), pca, model


def fit_hybrid_alternating(X_train, y_train, X_test, y_test, r=2,
                           lam=1.0, n_iter=40, random_state=0):
    rng = np.random.RandomState(random_state)
    n, p = X_train.shape
    # init U: top r PCA directions
    pca0 = PCA(n_components=r).fit(X_train)
    U = pca0.components_.T.copy()  # p x r
    alpha = np.zeros(r)
    for it in range(1, n_iter + 1):
        # 1) fix U, solve alpha by least squares
        Z = X_train @ U  # n x r
        alpha, *_ = np.linalg.lstsq(Z, y_train, rcond=None)
        # 2) fix alpha, update U by a gradient-like step on objective
        # objective: ||y - X U alpha||^2 + lam ||X - X U U^T||_F^2
        # residual for prediction term
        residual = (X_train @ U @ alpha) - y_train  # n
        grad_pred = (X_train.T @ residual)[:, None] @ alpha[None, :] * 2.0 / n
        # reconstruction gradient: 2 * lam * X^T (X U U^T - X) U / n
        XU = X_train @ U
        recon_term = X_train.T @ (XU @ U.T - X_train)  # p x p
        grad_recon = 2.0 * lam * (recon_term @ U) / n
        grad = grad_pred + grad_recon
        step = 1e-1 / it
        U = U - step * grad
        # re-orthonormalize columns to keep them stable
        U, _ = np.linalg.qr(U)
    y_pred = (X_test @ U) @ alpha
    return mean_squared_error(y_test, y_pred), U, alpha


def load_nba_features(csv_path, selected_features=None, dropna=True):
    # simple loader: read CSV, pick numeric columns
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    # data is structured array; convert to 2D numeric matrix of selected features
    header = data.dtype.names
    # if no features requested, take numeric columns after known non-numeric ones
    if selected_features is None:
        # guess numeric columns by trying to convert
        selected = []
        for name in header:
            try:
                _ = data[name].astype(float)
                selected.append(name)
            except Exception:
                pass
        selected_features = selected
    Xcols = []
    for name in selected_features:
        try:
            col = data[name].astype(float)
            Xcols.append(col)
        except Exception:
            pass
    X = np.vstack(Xcols).T
    if dropna:
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
    return X, selected_features


def run_synthetic_experiment(args):
    p = args.p
    Sigma = generate_X_sigma(p=p, random_state=args.seed)
    # We'll run a small sweep over lambda values and (optionally) angles
    lambdas = args.lambdas
    angles_deg = args.angles
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # If angles specified as degrees, construct beta by rotating between top and bottom EV
    evals, evecs = np.linalg.eigh(Sigma)
    v_top = evecs[:, np.argmax(evals)]
    v_bot = evecs[:, np.argmin(evals)]

    # Helper to build beta at angle t (in radians) from top EV towards bottom EV
    def beta_from_angle(t):
        # orthonormalize
        a = v_top / np.linalg.norm(v_top)
        b = v_bot - (a @ v_bot) * a
        if np.linalg.norm(b) < 1e-12:
            return a.copy()
        b = b / np.linalg.norm(b)
        beta = math.cos(t) * a + math.sin(t) * b
        beta /= np.linalg.norm(beta)
        return beta

    # If angles not provided, use 'small'/'large' choice
    if angles_deg is None:
        angles_deg = [0.0] if args.beta_dir == 'large' else [90.0]

    # Preallocate heatmap data
    heatmap = np.zeros((len(angles_deg), len(lambdas)))

    for i_ang, ang in enumerate(angles_deg):
        t = math.radians(ang)
        # construct beta direction at angle t
        if args.beta_dir in ('small', 'large') and angles_deg == [0.0, 90.0]:
            beta_dir_label = args.beta_dir
        beta_vec = beta_from_angle(t)

        # generate X once per angle but with samples
        rng = np.random.RandomState(args.seed + i_ang)
        X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=args.n)
        # scale noise to achieve SNR relative to chosen beta_vec
        signal_var = beta_vec.T @ Sigma @ beta_vec
        sigma_eps = np.sqrt(signal_var / args.snr)
        y = X @ beta_vec + rng.normal(scale=sigma_eps, size=args.n)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed + 2)

        # baseline methods (no lambda sweep)
        mse_ols, ols_model = fit_ols(X_train, y_train, X_test, y_test)
        mse_pcr, pca_model, pcr_model = fit_pcr(X_train, y_train, X_test, y_test, r=args.r)

        # sweep lambdas for hybrid
        hybrid_mses = []
        Us = []
        alphas = []
        for lam in lambdas:
            mse_h, U_est, alpha_est = fit_hybrid_alternating(
                X_train, y_train, X_test, y_test, r=args.r, lam=lam,
                n_iter=args.n_iter, random_state=args.seed + 3)
            hybrid_mses.append(mse_h)
            Us.append(U_est)
            alphas.append(alpha_est)
        hybrid_mses = np.array(hybrid_mses)
        heatmap[i_ang, :] = hybrid_mses

        # Save bar chart comparing MSEs (choose best hybrid across lambdas)
        best_idx = np.argmin(hybrid_mses)
        best_mse_h = hybrid_mses[best_idx]
        best_U = Us[best_idx]
        best_alpha = alphas[best_idx]

        labels = ['OLS', f'PCR (r={args.r})', f'Hybrid best (lam={lambdas[best_idx]})']
        vals = [mse_ols, mse_pcr, best_mse_h]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, vals, color=['C0', 'C1', 'C2'])
        plt.ylabel('Test MSE')
        plt.title(f'MSE comparison (angle={ang}°, snr={args.snr})')
        plt.tight_layout()
        plt.savefig(results_dir / f'mse_bar_angle{int(ang)}.png')
        plt.close()

        # Plot MSE vs lambda for hybrid
        plt.figure(figsize=(6, 4))
        plt.semilogx(lambdas, hybrid_mses, marker='o')
        plt.xlabel('lambda')
        plt.ylabel('Test MSE')
        plt.title(f'Hybrid MSE vs lambda (angle={ang}°)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f'mse_vs_lambda_angle{int(ang)}.png')
        plt.close()

        # Diagnostic: direction alignment between estimated U and true beta
        # compute cosine similarity between beta and each column of best_U
        cosines = (best_U.T @ beta_vec) / (np.linalg.norm(best_U, axis=0) * np.linalg.norm(beta_vec))
        plt.figure(figsize=(6, 3))
        plt.bar(range(1, len(cosines) + 1), cosines)
        plt.xlabel('component')
        plt.ylabel('cosine similarity with true beta')
        plt.title(f'Hybrid components alignment (angle={ang}°)')
        plt.tight_layout()
        plt.savefig(results_dir / f'hybrid_alignment_angle{int(ang)}.png')
        plt.close()

        print(f"Angle {ang}°: OLS={mse_ols:.4g}, PCR={mse_pcr:.4g}, Hybrid_best(lam={lambdas[best_idx]})={best_mse_h:.4g}")

    # Save heatmap across angles and lambdas
    plt.figure(figsize=(8, 4))
    im = plt.imshow(heatmap, aspect='auto', origin='lower',
                    extent=[np.log10(lambdas[0]), np.log10(lambdas[-1]),
                            (angles_deg[0] if angles_deg else 0), (angles_deg[-1] if angles_deg else 0)])
    plt.colorbar(im, label='Test MSE')
    plt.xlabel('log10(lambda)')
    plt.ylabel('angle (deg)')
    plt.title('Hybrid MSE heatmap')
    plt.tight_layout()
    plt.savefig(results_dir / f'hybrid_heatmap.png')
    plt.close()


def run_nba_experiment(args):
    csv_path = args.csv
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # choose a small set of numeric features (height, weight, wingspan, verticals, body fat, sprint)
    candidate = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach',
                 'Vertical (Max)', 'Vertical (No Step)', 'Weight', 'Body Fat', 'Sprint']
    X_all, feat_names = load_nba_features(csv_path, selected_features=candidate)
    # drop rows with NaNs already done
    p = X_all.shape[1]
    # synthetic response: choose beta aligned to smallest-eig direction of X covariance
    Sigma = np.cov(X_all, rowvar=False)
    X, y, beta_true, sigma_eps = sample_data(n=X_all.shape[0], p=p, Sigma=Sigma,
                                             beta_dir=args.beta_dir, snr=args.snr,
                                             random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=args.seed + 2)
    mse_ols, _ = fit_ols(X_train, y_train, X_test, y_test)
    mse_pcr, _, _ = fit_pcr(X_train, y_train, X_test, y_test, r=args.r)
    mse_hybrid, _, _ = fit_hybrid_alternating(X_train, y_train, X_test, y_test, r=args.r,
                                             lam=args.lam, n_iter=args.n_iter,
                                             random_state=args.seed + 3)
    print(f"Features used: {feat_names}")
    print(f"MSE OLS: {mse_ols:.5g}")
    print(f"MSE PCR (r={args.r}): {mse_pcr:.5g}")
    print(f"MSE Hybrid (r={args.r}, lam={args.lam}): {mse_hybrid:.5g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'nba'], default='synthetic')
    parser.add_argument('--csv', type=str, default='data/nba/nba_draft_combine_all_years.csv')
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--r', type=int, default=2)
    parser.add_argument('--lam', type=float, default=10.0)
    parser.add_argument('--n_iter', type=int, default=40)
    parser.add_argument('--beta_dir', choices=['small', 'large', 'random'], default='small')
    parser.add_argument('--snr', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.4)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--lambdas', type=float, nargs='*', default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument('--angles', type=float, nargs='*', default=None,
                        help='angles in degrees for sweeping beta direction; if omitted uses beta_dir')
    args = parser.parse_args()
    if args.mode == 'synthetic':
        run_synthetic_experiment(args)
    else:
        run_nba_experiment(args)


if __name__ == '__main__':
    main()
