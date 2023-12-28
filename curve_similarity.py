import numpy as np

from utils import EXP_LIST, get_random_seed

import traj_dist.distance as tdist

class FrechetDistance:
    def __init__(self):
        pass

    @staticmethod
    def calculate_euclid(point_a, point_b):
        """
        Args:
            point_a: a data point of curve_a
            point_b: a data point of curve_b
        Return:
            The Euclid distance between point_a and point_b
        """
        return np.linalg.norm(point_a - point_b, ord=2)

    @staticmethod
    def calculate_frechet_distance(dp, i, j, curve_a, curve_b):
        """
        Args:
            dp: The distance matrix
            i: The index of curve_a
            j: The index of curve_b
            curve_a: The data sequence of curve_a
            curve_b: The data sequence of curve_b
        Return:
            The frechet distance between curve_a[i] and curve_b[j]
        """
        if dp[i][j] > -1:
            return dp[i][j]
        elif i == 0 and j ==0:
            dp[i][j] = FrechetDistance.calculate_euclid(curve_a[0], curve_b[0])
        elif i > 0 and j == 0:
            dp[i][j] = max(
                FrechetDistance.calculate_frechet_distance(dp, i-1, 0, curve_a, curve_b),
                FrechetDistance.calculate_euclid(curve_a[i], curve_b[0])
            )
        elif i == 0 and j > 0:
            dp[i][j] = max(
                FrechetDistance.calculate_frechet_distance(dp, 0, j-1, curve_a,curve_b),
                FrechetDistance.calculate_euclid(curve_a[0],curve_b[j])
            )
        elif i > 0 and j > 0:
            dp[i][j] = max(min(
                FrechetDistance.calculate_frechet_distance(dp, i-1, j, curve_a, curve_b),
                FrechetDistance.calculate_frechet_distance(dp, i-1, j-1, curve_a, curve_b),
                FrechetDistance.calculate_frechet_distance(dp, i, j-1, curve_a, curve_b)),
                FrechetDistance.calculate_euclid(curve_a[i], curve_b[j])
            )
        else:
            dp[i][j] = float("inf")
        return dp[i][j]

    @staticmethod
    def get_similarity(curve_a, curve_b):
        dp = [[-1 for _ in range(len(curve_b))] for _ in range(len(curve_a))]
        similarity = FrechetDistance.calculate_frechet_distance(dp, len(curve_a)-1, len(curve_b)-1, curve_a, curve_b)
        return max(np.array(dp).reshape(-1, 1))[0]


def main():
    seed = get_random_seed()
    final_res = []
    distance_list = [
        tdist.sspd,
        tdist.dtw,
        tdist.lcss,
        tdist.hausdorff,
        tdist.discret_frechet,
        tdist.frechet,
        tdist.erp,
        tdist.edr
    ]
    for dynamic in ['separate', 'shallowdeep']:
        for backbone_id in range(3):
            for poisoning_rate in [0.05, 0.1, 0.15]:
                tmp_res = []
                for data_id in range(1):
                    exp = EXP_LIST[data_id * 3 + backbone_id]
                    backbone, dataset = exp
                    for dist_func in distance_list:
                        trace_path = './results/{}/curve/{}_{}_{}_{}.csv'.format(seed, dynamic, poisoning_rate, dataset, backbone)
                        trace = np.loadtxt(trace_path, delimiter=',')
                        t1 = trace[:, :2]
                        t2 = trace[:, 2:4]
                        t3 = trace[:, 4:6]
                        t4 = trace[:, 6:8]

                        s1 = dist_func(t1, t3)
                        tmp_res.append(s1)
                        s2 = dist_func(t1, t4)
                        tmp_res.append(s2)
                tmp_res = np.array(tmp_res).reshape([1, -1])
                final_res.append(tmp_res)
    final_res = np.concatenate(final_res)
    np.savetxt('./results/1221/sim.csv', final_res, delimiter=',')


if __name__ == '__main__':
    main()