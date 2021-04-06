# Based on code from PyTorch Speaker Verification:
# https://github.com/HarryVolek/PyTorch_Speaker_Verification
# Copyright (c) 2019, HarryVolek
# Additions Copyright (c) 2021, Jim O'Regan
# License: MIT
import numpy as np

# wav2vec2's max duration is 40 seconds, using 39 by default
# to be a little safer
def concat(times, segs, max_duration=39.0):
    """
    Concatenate continuous times and their segments, where the end time
    of a segment is the same as the start time of the next
        Parameters:
            times: list of tuple (start, end)
            segs: list of segments (audio frames)
            max_duration: maximum duration of the resulting concatenated
                segments; the kernel size of wav2vec2 is 40 seconds, so
                the default max_duration is 39, to ensure the resulting
                list of segments will fit
        Returns:
            concat_times: list of tuple (start, end)
            concat_segs: list of segments (audio frames)
    """
    absolute_maximum=40.0
    if max_duration > absolute_maximum:
        raise Exception('`max_duration` {:.2f} larger than kernel size (40 seconds)'.format(max_duration))
    # we take 0.0 to mean "don't concatenate"
    do_concat = (max_duration != 0.0)
    concat_seg = []
    concat_times = []
    seg_concat = segs[0]
    time_concat = times[0]
    for i in range(0, len(times)-1):
        can_concat = (times[i+1][1] - time_concat[0]) < max_duration
        if time_concat[1] == times[i+1][0] and do_concat and can_concat:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
            time_concat = (time_concat[0], times[i+1][1])
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
            concat_times.append(time_concat)
            time_concat = times[i+1]
    else:
        concat_seg.append(seg_concat)
        concat_times.append(time_concat)
    return concat_times, concat_seg