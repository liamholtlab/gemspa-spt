import gemspa_spt

test_folder = "../test_data"
test_results = f"{test_folder}/test_results"


if __name__ == '__main__':

    print("mosaic")
    pt = gemspa_spt.ParticleTracks(path=f"{test_folder}/Mosaic_Results.csv",
                                   data_format="mosaic")
    track_id = pt.track_ids[1]
    print(track_id)

    msds = pt.msd(track_id)
    print(msds)

    step_sizes = pt.step_size(track_id)
    print(step_sizes)

    print("R of G")
    rg = pt.r_of_g(track_id, full=False)
    print(rg)

    rg = pt.r_of_g(track_id, full=True)
    print(rg)

    res = pt.fit_msd_linear(msds[1:, 0], msds[1:, 4], 2, max_lagtime=10, err=True)
    print(res)
    res = pt.fit_msd_linear(msds[1:, 0], msds[1:, 4], 2, max_lagtime=10, err=False)
    print(res)

    res = pt.fit_msd_loglog(msds[1:, 0], msds[1:, 4], 2, max_lagtime=10)
    print(res)

    res = pt.step_size_all_tracks()
    print(len(res))

    res = pt.msd_all_tracks()
    print(len(res))

    res = pt.r_of_g_all_tracks(full=True)
    print(res)

    res = pt.r_of_g_all_tracks(full=False)
    print(res[:10])

    res = pt.ensemble_avg_msd()
    print(res)

    res = pt.fit_msd_all_tracks(linear_fit=True, all_dims=False)
    print(res[:10])
    print(len(res))
    res = pt.fit_msd_all_tracks(linear_fit=False, all_dims=False)
    print(len(res))

    res = pt.fit_msd_all_tracks(linear_fit=True, all_dims=True)
    print(res[:10])
    print(len(res))
    res = pt.fit_msd_all_tracks(linear_fit=False, all_dims=True)
    print(len(res))


