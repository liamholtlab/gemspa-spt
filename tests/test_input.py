import gemspa_spt

test_folder = "../test_data"
test_results = f"{test_folder}/test_results"


def print_results(obj):
    print(obj.tracks_df)
    print(obj.track_lengths[:10])
    print(obj.track_ids[:10])
    print(obj.tracks[:10])
    print(obj.dimension)


def write_files(obj, loc_root):
    obj.write_track_file(f"{loc_root}.csv", sep=',')
    obj.write_track_file(f"{loc_root}.txt", sep='\t')


if __name__ == '__main__':

    print("trackmate")
    pt = gemspa_spt.ParticleTracks(path=f"{test_folder}/trackmate_tracks.csv",
                                   data_format="trackmate")
    print_results(pt)
    write_files(pt, f"{test_results}/trackmate_tracks_output")

    #####
    print("trackmate from data frame")
    pt = gemspa_spt.ParticleTracks(tracks_df=pt.tracks_df,
                                   data_format="trackmate")
    print_results(pt)
    write_files(pt, f"{test_results}/trackmate_tracks_output2")

    #####
    print("mosaic")
    pt = gemspa_spt.ParticleTracks(path=f"{test_folder}/Mosaic_Results.csv",
                                   data_format="mosaic")
    print_results(pt)
    write_files(pt, f"{test_results}/mosaic_tracks_output")

    #####
    print("gemspa")
    pt = gemspa_spt.ParticleTracks(path=f"{test_folder}/gemspa_tracks.txt",
                                   data_format="gemspa", sep='\t')
    print_results(pt)
    write_files(pt, f"{test_results}/gemspa_tracks_output")

    #####
    print("gemspa from numpy array")
    pt = gemspa_spt.ParticleTracks(tracks=pt.tracks,
                                   data_format="gemspa")
    print_results(pt)
    write_files(pt, f"{test_results}/gemspa_tracks_output2")

    #####
    print("trackpy")
    pt = gemspa_spt.ParticleTracks(path=f"{test_folder}/trackpy_tracks.txt",
                                   data_format="trackpy", sep='\t')
    print_results(pt)
    write_files(pt, f"{test_results}/trackpy_tracks_output")

