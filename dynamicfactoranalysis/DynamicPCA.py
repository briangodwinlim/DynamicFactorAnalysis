"""
Dynamic PCA

Author: Brian Godwin Lim
"""

import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import partial

class DynamicPCA:
    """
    Dynamic Principal Component Analysis

    Parameters
    ----------
    data : array_like
        nobs x nvars data matrix where rows are observations 
        and columns are features.
    dpcs : int, optional
        Number of dynamic principal components to calculate.
        The parameter ncomp may be used as an alternative (statsmodels PCA).
    M : {int, None}, optional
        Number of leads and lags to include in spectral density matrix estimation.
        If None, M is the value in Forni et al. (2000). 
    q_leads : int, optional
        Number of leads in dynamic principal components to include in 
        projection of common components.
    p_lags : int, optional
        Number of lags in dynamic principal components to include in 
        projection of common components.
    demean : bool, optional
        Flag indicating whether to demean data before computing dynamic principal
        components. 
    standardize : bool, optional
        Flag indicating whether to standardize data before computing
        dynamic principal components. If True, demean is automatically True.
    normalize : bool, optional
        Flag indicating whether to normalize the dynamic principal components 
        to have unit inner product.
    weights : ndarray, optional
        nvars array of weights to use after transforming data according to standardize
        or demean when computing the dynamic principal components.
    missing : {str, None}, optional
        Method for adjusting missing data. Choices are:
            * 'drop-row' - drop rows with missing values.
            * 'drop-col' - drop columns with missing values.
            * 'drop-min' - drop either rows or columns, choosing by data retention.
            * 'fill-em' - use EM algorithm to fill missing values.  
                dpcs should be set to the number of principal components required.
            * `None` raises ValueError if data contains NaN or inf values.
    tol_em : float, optional
        Tolerance to use when checking for convergence of the EM algorithm.
    max_em_iter : int, optional
        Maximum iterations for the EM algorithm.
    resample : bool, optional
        Resample and fill forward after cleaning data.
        Works only if data is a DataFrame.

    Attributes
    ----------
    filter : ndarray
        (2M + 1) x nvars arrays of two-sided filters 
        for each dynamic principal component.
    factors : array or DataFrame
        (nobs - 2M) x dpcs centered array of dynamic principal components.
    common : array or DataFrame
        (nobs - 2M - q - p) x nvars centered array of fitted common components.
    rsquared : Series
        (dpcs + 1) array where the ith element is the R-squared
        of including the first i dynamic principal components. 
        Note: values are calculated on the transformed data, not the original data.
    transformed_data : array or DataFrame
        Demeaned, standardized, and weighted data used to compute
        principal components and related quantities.
    weights : ndarray
        nvars array of weights used to compute the dynamic principal components,
        normalized to unit length.

    References
    -----
        Forni et al. (2000) 
        Forni et al. (2005) 
        Favero et al. (2005)
    """

    def __init__(self, data, dpcs=2, M=None, q_leads=0, p_lags=0, 
                 demean=True, standardize=True, normalize=True, weights=None, 
                 missing=None, tol_em=5e-8, max_em_iter=100, resample=True, **kwargs):
        
        # DPCA hyperparameters
        self.dpcs = self.ncomp = kwargs.get('ncomp', dpcs)
        self.M = round(2 / 3 * np.power(data.shape[0], 1 / 3)) if M is None else M   # Forni et al (2000)
        self.q_leads = q_leads
        self.p_lags = p_lags
        if self.dpcs > min(data.shape):
            raise ValueError('The requested number of components is more than can be computed from data.')

        # Data transformation parameters
        self.demean = demean or standardize
        self.scale = standardize
        self.normalize = normalize
        self._missing = missing
        self._tol_em = tol_em
        self._max_em_iter = max_em_iter
        self.resample = resample
        self.weights = np.ones(data.shape[1]) if weights is None else np.asarray(weights).flatten()
        self.weights = self.weights / np.sqrt(np.mean(self.weights ** 2.0))
        if self.weights.size != data.shape[1]:
            raise ValueError('weights should be of size nvars.')
        
        self.kwargs = kwargs
        
        # Model fitting
        self._fitted = False
        self._proj_mod = None
        self._mu, self._sigma = (0, 1)
        self.data = data
        self.process_data()
        self.fit()
    
    def process_data(self):
        self._index = getattr(self.data, 'index', None)
        self._columns = getattr(self.data, 'columns', None)

        self._transformed_data = np.asarray(self.data)
        self._transformed_data = self._adjust_missing()
        self._transformed_data = self._prepare_data()
        self.transformed_data = self._to_pandas(self._transformed_data)

        if self.resample and self._index is not None:
            if getattr(self._index, 'freq', False):
                self.transformed_data = self.transformed_data.resample(self._index.freq).last().ffill()
                self._transformed_data = np.asarray(self.transformed_data)
                self._index = self.transformed_data.index
                
        self.T, self.N = self.transformed_data.shape
    
    def _adjust_missing(self):
        def keep_row(x):
            index = np.logical_not(np.isnan(x).any(axis=1))
            return x[index, :], index

        def keep_col(x):
            index = np.logical_not(np.isnan(x).any(axis=0))
            return x[:, index], index

        data = np.copy(self._transformed_data)
        _index, _columns, weights = self._index, self._columns, self.weights
        
        if self._missing == 'drop-row':
            data, index = keep_row(data)
            _index = _index[index] if _index is not None else None 
        elif self._missing == 'drop-col':
            data, index = keep_col(data)
            _columns = _columns[index] if _columns is not None else None 
            weights = weights[index]
        elif self._missing == 'drop-min':
            drop_row, drop_row_index = keep_row(data)
            drop_col, drop_col_index = keep_col(data)

            if drop_row.size > drop_col.size:
                data = drop_row
                _index = _index[drop_row_index] if _index is not None else None 
            else:
                data = drop_col
                _columns = _columns[drop_col_index] if _columns is not None else None 
                weights = weights[drop_col_index]
        elif self._missing == 'fill-em':
            data = self._fill_missing_em()
        elif self._missing is None:
            if not np.isfinite(data).all():
                raise ValueError('Data contains non-finite values (inf, NaN).'
                                 'Use one of the methods for adjusting missing values.')
        else:
            raise ValueError('missing method is not known.')
        
        if data.size == 0:
            raise ValueError('Removal of missing values has eliminated all data.')
        
        self._index, self._columns, self.weights = _index, _columns, weights
        return data
    
    def _fill_missing_em(self):
        def _norm(x):
            return np.sqrt(np.sum(x * x))
        
        def _fill_missing(data):
            if self.kwargs.get('fill_em_mean', False):
                return np.ones((data.shape[0], 1)) * np.nanmean(data, axis=0)
            else:
                ffill = self._ffill(data)
                bfill = self._bfill(data)
                return (np.nan_to_num(ffill) + np.nan_to_num(bfill)) / (2 - np.isnan(ffill) - np.isnan(bfill))

        def _em_step(data, mask, mask_edge):
            self._transformed_data = np.copy(data)
            projection = np.asarray(self.project(transform=False, unweight=False, dropna=False))
            
            projection_masked = projection[mask]
            data[mask] = projection_masked
            data[mask_edge] = np.nan                # Remove filled data for M leads and lags
            data[mask_edge] = _fill_missing(data)[mask_edge]

            return data, projection_masked

        # 1. Check for nans
        non_missing = np.logical_not(np.isnan(self._transformed_data))
        
        if np.all(non_missing):
            return self._transformed_data
        
        col_non_missing = np.sum(non_missing, axis=1)
        row_non_missing = np.sum(non_missing, axis=0)
        if np.any(col_non_missing < self.dpcs) or np.any(row_non_missing < self.dpcs):
            raise ValueError('Implementation requires that all columns and '
                             'all rows have at least dpcs non-missing values')

        # 2. Standardize data as needed
        data = self._prepare_data()

        # 3. Get nan mask
        mask_all = np.isnan(data)

        # 4. Replace missing with filled data
        projection = _fill_missing(data)
        data[mask_all] = projection[mask_all]

        # 5. Update mask for M leads and lags
        mask = np.copy(mask_all)
        mask[:min(self.M, data.shape[0])] = False
        mask[max(0, data.shape[0] - self.M):] = False
        projection_masked = projection[mask]
        mask_edge = np.logical_xor(mask_all, mask)

        # 6. Compute eigenvalues and fit
        diff = 1.0
        _iter = 0
        while diff > self._tol_em and _iter < self._max_em_iter and np.sum(mask):
            last_projection_masked = projection_masked
            data, projection_masked = _em_step(data, mask, mask_edge)

            delta = last_projection_masked - projection_masked
            diff = _norm(delta) / _norm(projection_masked)
            _iter += 1
        
        data, projection_masked = _em_step(data, mask, mask_edge)
        data = data * self._sigma + self._mu
        return data
    
    def _prepare_data(self):
        data = np.copy(self._transformed_data)
        if self.demean:
            self._mu = np.nanmean(data, axis=0) if not self._fitted else self._mu
            data = data - self._mu
        if self.scale:
            self._sigma = (np.clip(np.nanstd(data, axis=0), a_min=self.kwargs.get('sigma_min', 0.0), a_max=np.inf)
                           if not self._fitted else self._sigma)
            data = data / self._sigma
        return data / np.sqrt(self.weights)

    def _nancov(self, x, y):
        xx = np.nan_to_num(x)
        yy = np.nan_to_num(y)
        mask_x = np.logical_not(np.isnan(x)) * 1.0
        mask_y = np.logical_not(np.isnan(y)) * 1.0
        return np.dot(xx.T, yy) / (np.dot(mask_x.T, mask_y) - 1)

    # Sample covariance matrix \Gamma(k)
    def covariance_matrix(self, k, data):
        data = np.asarray(data)
        # data = data - np.mean(data, axis=0)
        T, _ = data.shape
        Xt0 = data[max(0, k)  : min(T, T + k)]
        Xtk = data[max(0, -k) : min(T, T - k)]
        return self._nancov(Xt0, Xtk)

    @property
    def k_range(self):
        return np.arange(-self.M, self.M + 1)
    
    @property
    def _h(self):
        return np.arange(0, 2 * self.M + 1)

    @property
    def omegas(self):
        return 2 * np.pi * self._h / self._h.size
    
    def _w(self, k):
        return 1 - np.abs(k) / (self.M + 1)
    
    # Spectral density matrix \Sigma(\omega_h)
    def spectral_density_matrix(self, omega_h, covariance_matrix):
        return np.sum([
            self._w(k) * 
            np.exp(-1j * k * omega_h) * 
            covariance_matrix(k)
        for k in self.k_range], axis=0) / (2 * np.pi)
    
    def _spectral_data(self, omega_h):
        covariance_matrix = partial(self.covariance_matrix, data=self._transformed_data)
        return self.spectral_density_matrix(omega_h, covariance_matrix)

    # Eigenvectors p_j(\omega_h)
    def eigvecs(self, omega_h, spectral_density_matrix):
        eigvals, eigvecs = scipy.linalg.eig(spectral_density_matrix(omega_h))
        return eigvecs[:, np.argsort(eigvals)[::-1]]
    
    def _eigvecs_data(self, omega_h):
        return self.eigvecs(omega_h, self._spectral_data)

    # Two-sided filter \lambda_j(L)
    @property
    def filter(self):
        if not self._fitted:
            eigvecs = np.stack([self._eigvecs_data(omega_h) for omega_h in self.omegas])
            fourier = np.exp(1j * self.k_range.reshape(1,-1) * self.omegas.reshape(-1,1))
            self._filter = np.mean(fourier[:,:,np.newaxis,np.newaxis] * eigvecs[:,np.newaxis,:,:], axis=0)
        assert np.isclose(self._filter[:,:,:self.dpcs].imag, 0, atol=1e-5).all()
        return self._filter[:,:,:self.dpcs].real

    # Dynamic principal components f_t
    @property
    def _factors(self):
        dynamic_pcs = np.array([scipy.signal.correlate(self._transformed_data, self.filter[:,:,j], mode='valid').flatten()
                                for j in range(self.dpcs)]).T
        scale = np.sqrt(np.nansum(dynamic_pcs ** 2.0, axis=0)) if self.normalize else 1
        return self._padna(dynamic_pcs / scale, before=self.M, after=self.M)
    
    @property
    def factors(self):
        return self._dropna(self._to_pandas(self._factors, columns=[f'DPC{i}' for i in range(1, self.dpcs + 1)]))

    # Common component \chi_t
    def project(self, use_fitted=False, dpcs=None, q_leads=None, p_lags=None, transform=True, unweight=True, dropna=True):
        """
        Compute the orthogonal projection of dynamic principal components to data
        using ordinary least squares.

        Parameters
        ----------
        use_fitted : bool, optional
            Flag indicating whether to use fitted projection.
            The values of dpcs, q_leads, p_lags are from the model specification.
        dpcs : {int, None}, optional
            Number of dynamic principal components to use. 
            If None, dpcs in model specification is used.
        q_leads : {int, None}, optional
            Number of leads in dynamic principal components to include.
            If None, q_leads in model specification is used.
        p_lags : {int, None}, optional
            Number of lags in dynamic principal components to include.
            If None, p_lags in model specification is used.
        transform : bool, optional
            Flag indicating whether to return the projection in the original
            space of the data or in the space of the demeaned/standardized/weighted data.
        unweight : bool, optional
            Flag indicating whether to undo the effects of the estimation weights.
        dropna : bool, optional
            Flag indicating whether to drop nans in the first (M + p) and last (M + q) observations. 

        Returns
        -------
        common : array or DataFrame
            (nobs - 2M - q - p) x nvars centered array of fitted common components if dropna is True.
            nobs x nvars centered array of fitted common components if dropna is False.
        """

        dpcs = self.dpcs if dpcs is None or use_fitted else dpcs
        q_leads = self.q_leads if q_leads is None or use_fitted else q_leads
        p_lags = self.p_lags if p_lags is None or use_fitted else p_lags
        if dpcs > self.dpcs:
            raise ValueError('dpcs must not exceed the number of components computed.')
        
        exog = np.hstack([self._shift(self._factors, periods=p, pad=np.nan)
                          for p in np.arange(-q_leads, p_lags + 1)])
        exog[:, dpcs:] = 0

        proj_mod = self._proj_mod if use_fitted else sm.OLS(self._transformed_data, exog, missing='drop').fit()
        common = np.asarray(proj_mod.predict(exog))

        common = common * np.sqrt(self.weights) if transform or unweight else common
        common = common * self._sigma + self._mu if transform else common
        common = self._to_pandas(common)
        return self._dropna(common) if dropna else common
    
    def fit(self):
        exog = np.hstack([self._shift(self._factors, periods=p, pad=np.nan)
                                for p in np.arange(-self.q_leads, self.p_lags + 1)])
        proj_mod = sm.OLS(self._transformed_data, exog, missing='drop').fit()
        self._proj_mod = proj_mod if not self._fitted else self._proj_mod
        self.common = self.project(use_fitted=True)
        self._fitted = True

    @property
    def rsquared(self):
        # np.nanmean to normalize due to missing observations in projection
        data = self._transformed_data * np.sqrt(self.weights)
        self._tss_indiv = np.nanmean(data ** 2, axis=0)
        self._tss = np.nanmean(self._tss_indiv)
        self._ess = np.zeros(self.dpcs + 1)
        self._ess_indiv = np.zeros((self.dpcs + 1, self.N))
        for i in range(self.dpcs + 1):
            projection = self.project(dpcs=i, transform=False, unweight=False)
            rss_indiv = np.nanmean(projection ** 2, axis=0)
            rss = np.nanmean(rss_indiv)
            self._ess[i] = self._tss - rss
            self._ess_indiv[i,:] = self._tss_indiv - rss_indiv
        rsquared = 1.0 - self._ess / self._tss
        rsquared = pd.Series(rsquared, name='rsquared')
        rsquared.index.name = 'dpcs'
        return rsquared

    # Inverse Fourier transform of spectral density matrix
    def inverse_spectral_density_matrix(self, k, spectral_density_matrix):
        return np.mean([
            np.exp(1j * k * omega_h) * 
            spectral_density_matrix(omega_h)
        for omega_h in self.omegas], axis=0)
    
    # Forecasting
    def forecast(self, steps=1, use_fitted=False, dpcs=None, q_leads=None, p_lags=None, r_gpcs=None, transform=True):
        """
        Forecasting using dynamic principal components and projected common components.

        Parameters
        ----------
        steps : int, optional
            Number of periods from the last observation to forecast.
        use_fitted : bool, optional
            Flag indicating whether to use fitted projection.
            The values of dpcs, q_leads, p_lags are from the model specification.
        dpcs : {int, None}, optional
            Number of dynamic principal components to use. 
            If None, dpcs in model specification is used.
        q_leads : {int, None}, optional
            Number of leads in dynamic principal components to include.
            If None, q_leads in model specification is used.
        p_lags : {int, None}, optional
            Number of lags in dynamic principal components to include.
            If None, p_lags in model specification is used.
        r_gpcs : {int, None}, optional
            Number of generalized principal components to include 
            in forecasting as in Forni et al. (2005). If None, dpcs is used.
        transform : bool, optional
            Flag indicating whether to return the projection in the original
            space of the data or in the space of the demeaned/standardized/weighted data.

        Returns
        -------
        predicted : array or DataFrame
            steps x nvars array of forecasted values.
        """

        def real_covariance(k, data):
            covariance_matrix = partial(self.covariance_matrix, data=data)
            spectral_density_matrix = partial(self.spectral_density_matrix, covariance_matrix=covariance_matrix)
            fourier_covariance_matrix = partial(self.inverse_spectral_density_matrix, spectral_density_matrix=spectral_density_matrix)
            gamma = fourier_covariance_matrix(k)
            assert np.isclose(gamma.imag, 0, atol=1e-5).all()
            return gamma.real
        
        r_gpcs = self.dpcs if r_gpcs is None else r_gpcs
        dpcs = self.dpcs if dpcs is None or use_fitted else dpcs
        q_leads = self.q_leads if q_leads is None or use_fitted else q_leads
        p_lags = self.p_lags if p_lags is None or use_fitted else p_lags
        if dpcs > self.dpcs:
            raise ValueError('dpcs must not exceed the number of components computed.')
        
        common = self.project(use_fitted=use_fitted, dpcs=dpcs, q_leads=q_leads, p_lags=p_lags, transform=False, dropna=False)
        resid = self._transformed_data - common
        Gamma_chi = partial(real_covariance, data=common)
        Gamma_eps = partial(real_covariance, data=resid)

        eigvals, eigvecs = scipy.linalg.eig(Gamma_chi(0), Gamma_eps(0))
        eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
        lambda_h = eigvecs[:, :r_gpcs]
        xT = self._transformed_data[-1:,:].T

        common_term = lambda_h.dot(np.linalg.inv(lambda_h.T.dot(Gamma_chi(0)).dot(lambda_h))).dot(lambda_h.T).dot(xT)
        predicted = np.array([Gamma_chi(h).dot(common_term).flatten() for h in range(1, steps + 1)])
        predicted = predicted * np.sqrt(self.weights) * self._sigma + self._mu if transform else predicted
        return self._to_pandas(predicted, index=[self.transformed_data.index[-1] + h for h in range(1, steps + 1)])

    def apply(self, data, refit=False):
        """
        Apply the fitted parameters to new data unrelated to the original data

        Parameters
        ----------
        data : array or DataFrame
            New observations from the modeled time-series process.
        refit : bool, optional
            Whether to re-fit the parameters based on the new data.

        Notes
        -----
        The `data` argument to this method should consist of new observations
        that are not necessarily related to the original model's `data`.
        For data that continues the original data, see the `append` method.

        When re-fitting the model, the value of M follows the original model. 
        To specify a different value, create a new model.
        """

        self._fitted = not refit
        self.data = data
        self.process_data()
        self.fit()
    
    def append(self, data, refit=False):
        """
        Updates the model with new data appended to the original data

        Parameters
        ----------
        data : array or DataFrame
            New observations from the modeled time-series process.
        refit : bool, optional
            Whether to re-fit the parameters based on the combined data.

        Notes
        -----
        The `data` argument to this method should consist of new observations
        that occurred directly after the original model's `data`. 
        For any other kind of data, see the `apply` method.

        When re-fitting the model, the value of M follows the original model. 
        To specify a different value, create a new model.
        """

        self.apply(self._append(self.data, data), refit=refit)

    def __repr__(self):
        return f'Dynamic Principal Component Analysis(dpcs={self.dpcs}, M={self.M}, id={id(self)})'

    # Convert array to Dataframe if possible
    def _to_pandas(self, array, index=None, columns=None):
        if isinstance(self.data, pd.DataFrame):
            index = self._index if index is None else index
            columns = self._columns if columns is None else columns
        return (array if index is None or columns is None else 
                pd.DataFrame(array, index=index, columns=columns))
    
    # Shift array
    def _shift(self, array, periods=0, axis=0, pad=0):
        if isinstance(array, pd.DataFrame):
            return array.shift(periods=periods, axis=axis, fill_value=pad)
        else:
            array_ = np.swapaxes(array, 0, axis)
            array_ = np.roll(array_, periods, axis=0)
            array_[:max(0, periods)] = pad
            array_[max(0, array.shape[axis] + periods):] = pad
            array_ = np.swapaxes(array_, 0, axis)
            return array_

    # Row-level padna
    def _padna(self, array, before=0, after=0):
        if before < 0:
            raise ValueError('before should be non-negative.')
        if after < 0:
            raise ValueError('after should be non-negative.')
        
        if isinstance(array, pd.DataFrame):
            return pd.concat([
                pd.DataFrame(np.full((before, array.shape[1]), np.nan), 
                             index=[array.index[0] - h for h in range(1, before + 1)], 
                             columns=array.columns),
                array,
                pd.DataFrame(np.full((after, array.shape[1]), np.nan), 
                             index=[array.index[-1] + h for h in range(1, after + 1)], 
                             columns=array.columns)
            ])
        else:
            return np.pad(array, ((before, after), (0, 0)), constant_values=np.nan)

    # Row-level dropna
    def _dropna(self, array):
        if isinstance(array, pd.DataFrame):
            return array.dropna()
        else:
            return array[np.logical_not(np.isnan(array)).any(axis=1)]
    
    # Row-level forward fill
    def _ffill(self, array):
        if isinstance(array, pd.DataFrame):
            return array.ffill()
        else:
            idx = np.where(np.logical_not(np.isnan(array)).T, np.arange(array.shape[0]), 0).T
            idx = np.maximum.accumulate(idx, axis=0)
            return array[idx, np.arange(array.shape[1]).reshape(1,-1)]

    # Row-level backward fill
    def _bfill(self, array):
        return self._ffill(array[::-1])[::-1]

    # Row-level append array2 to array1
    def _append(self, array1, array2):
        if array1.shape[1] != array2.shape[1]:
            raise ValueError('array1 and array2 should have the same number of columns.')
        
        if isinstance(array1, pd.DataFrame) and isinstance(array2, pd.DataFrame):
            if getattr(array1.index, 'freq', None) != getattr(array2.index, 'freq', None):
                raise ValueError('array2 does not have an index that extends the index of array1.')
            if array1.index[-1] >= array2.index[0]:
                raise ValueError('index of array2 overlaps with index of array1')
            return pd.concat([array1, array2]).sort_index()
        else:
            return np.concatenate([np.asarray(array1), np.asarray(array2)])
    