import numpy as np
import math
import copy

from .utils import sliding_window, z_normalize
from .metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)

    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone

    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start: zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile

    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match

    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    }

    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder

    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r

    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone

        Parameters
        ----------
        m: length of subsequence

        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone

    def perform(self):
        raise NotImplementedError


class NaiveBestMatchFinder(BestMatchFinder):
    """
    Наивный поиск best-match с DTW:
    - корректная работа и для 1D ts (сам строит окна), и для 2D windows
    - быстрая нормализация, минимизация копий/аллокаций
    """

    def __init__(self, excl_zone_frac: float = 1.0, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)

    def _rolling_mean_std(self, ts: np.ndarray, m: int, eps: float = 1e-8):
        """
        Быстрые скользящие mean/std для 1D ряда.
        Возвращает два вектора длины N = n - m + 1.
        """
        ts = ts.astype(np.float64, copy=False)
        csum = np.concatenate(([0.0], np.cumsum(ts)))
        csum2 = np.concatenate(([0.0], np.cumsum(ts * ts)))
        win_sum = csum[m:] - csum[:-m]
        win_sum2 = csum2[m:] - csum2[:-m]
        mu = win_sum / m
        var = np.maximum(win_sum2 / m - mu * mu, 0.0)
        sigma = np.sqrt(var) + eps
        return mu.astype(np.float32), sigma.astype(np.float32)

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Ищет top-K подпоследовательностей временного ряда, наиболее близких к запросу по DTW.
        Возвращает словарь {'index': [...], 'distance': [...]}.
        """
        # Подготовка данных
        if ts_data.ndim == 1:
            m = int(len(query))
            windows = sliding_window(ts_data, m)      # (N, m), view без копии
            ts1d = ts_data
        elif ts_data.ndim == 2:
            windows = ts_data
            m = windows.shape[1]
            ts1d = None
        else:
            raise ValueError("ts_data must be 1D (raw series) or 2D (windows).")

        N, m_chk = windows.shape
        assert m_chk == m

        excl_zone = self._calculate_excl_zone(m)
        eps = 1e-8

        # Нормализация запроса один раз
        if self.is_normalize:
            q = z_normalize(query.astype(np.float32, copy=False))
        else:
            q = query.astype(np.float32, copy=False)

        # Нормализация окон
        if self.is_normalize:
            if ts1d is not None:
                # Быстрые скользящие mean/std для 1D
                mu, sigma = self._rolling_mean_std(ts1d, m, eps=eps)
            else:
                # 2D вход: посчитаем статистики векторно (быстро в NumPy)
                mu = windows.mean(axis=1, dtype=np.float64).astype(np.float32)
                # var = E[x^2] - mu^2 (стабильнее, чем (x-mu)^2.mean())
                ex2 = (windows.astype(np.float32, copy=False) ** 2).mean(axis=1, dtype=np.float64).astype(np.float32)
                var = np.maximum(ex2 - (mu.astype(np.float64) ** 2).astype(np.float32), 0.0)
                sigma = np.sqrt(var) + eps
        else:
            mu = sigma = None

        # Основной проход
        dist_profile = np.full(N, np.inf, dtype=np.float32)
        bsf = np.inf

        # NB: если твоя DTW реализация поддерживает upper bound (early abandoning),
        # можно передать bsf и ускорить. Здесь используем интерфейс как есть.
        for i in range(N):
            subseq = windows[i].astype(np.float32, copy=False)

            if self.is_normalize:
                s = sigma[i]
                if not np.isfinite(s) or s <= 0.0:
                    # окно константное — пропускаем
                    continue
                subseq = (subseq - mu[i]) / s

            d = DTW_distance(q, subseq)
            dist_profile[i] = d
            if d < bsf:
                bsf = d

        # Выбор top-K c учётом exclusion zone
        topK_results = topK_match(dist_profile, excl_zone, self.topK)

        return {
            'index': topK_results['indices'],
            'distance': topK_results['distances'],
        }


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder

    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0

    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences

        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence

        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        lb_Kim = 0

        lb_Kim = (subs1[0] - subs2[0]) ** 2 + (subs1[-1] - subs2[-1]) ** 2

        return lb_Kim

    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences

        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size

        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        lb_Keogh = 0

        n = len(subs1)
        lb_Keogh = 0.0

        # Compute the envelopes
        u = np.zeros(n)
        l = np.zeros(n)
        for i in range(n):
            window_start = max(0, i - r)
            window_end = min(n, i + r + 1)
            u[i] = max(subs1[window_start:window_end])
            l[i] = min(subs1[window_start:window_end])

        # Calculate LB_Keogh as the sum of squared differences for points outside the envelopes
        for i in range(n):
            if subs2[i] > u[i]:
                lb_Keogh += (subs2[i] - u[i]) ** 2
            elif subs2[i] < l[i]:
                lb_Keogh += (subs2[i] - l[i]) ** 2

        return lb_Keogh

    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series

        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm

        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2):  # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,)) * np.inf
        bsf = np.inf

        bestmatch = {
            'index': [],
            'distance': []
        }

        for start in range(N):
            subsequence = ts_data[start]
        if self.is_normalize:
            subsequence = z_normalize(subsequence)

            # Apply lower bound LB_Kim
        lb_Kim = self._LB_Kim(query, subsequence)
        if lb_Kim < bsf:
            # Apply lower bound LB_Keogh (QC)
            lb_Keogh_QC = self._LB_Keogh(query, subsequence, self.r)
            if lb_Keogh_QC < bsf:
                # Apply lower bound LB_Keogh (CQ)
                lb_Keogh_CQ = self._LB_Keogh(subsequence, query, self.r)
                if lb_Keogh_CQ < bsf:
                    # Calculate DTW distance
                    distance = DTW_distance(query, subsequence)
                    if distance < bsf:
                        dist_profile[start] = distance
                        bsf = distance

        # Extract top-K matches
        topK_results = topK_match(dist_profile, excl_zone, self.topK)
        bestmatch['index'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']

        return bestmatch