#!/usr/bin/env python
'''
Copyright (C) 2017 Gabor Papp
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

'''
import argparse
import glob
import json
import sys

frames = []
id_counter = 1

class Frame:
    def __init__(self):
        self.poses = []

    def add_pose(self, pose):
        self.poses.append( pose )

    def debug(self):
        for p in self.poses:
            print p.keypoints,
        print

    poses = []

class Pose:
    def __init__(self, keypoints):
        self.keypoints = [keypoints[i:i + 3] for i in xrange(0, len(keypoints), 3)]

    def get_flat_keypoints(self):
        return [i for sublist in self.keypoints for i in sublist]

    def distance_squared(self, other):
        d = 0.0
        for i in range(0, len(self.keypoints)):
            # only calculate distance for points with suitable confidence
            if (self.keypoints[i][2] > 0.1 and other.keypoints[i][2] > 0.1):
                d += (self.keypoints[i][0] - other.keypoints[i][0]) ** 2 + \
                     (self.keypoints[i][1] - other.keypoints[i][1]) ** 2
            else:
                d += 1000.0 # otherwise punish a fixed error
        return d

    def is_seen(self):
        return self.seen

    id = -1
    seen = True

def load_json_folder(keypoints_folder):
    files = sorted(glob.glob(keypoints_folder + "/*.json"))
    global frames
    for f in files:
        with open(f) as data_file:
            data = json.load(data_file)
        frame = Frame()
        for person in data["people"]:
            frame.add_pose(Pose(person["pose_keypoints"]))
        frames.append(frame)

'''
    Returns the closest pose id if found or -1
'''
def find_closest_knn(pose, new_poses, k):
    winner = -1
    # list of neighbour point index and respective distances
    nbors = []

    for i in range(len(new_poses)):
        new_pose = new_poses[i]
        d_sq = pose.distance_squared(new_pose)

        # check if this blob is closer to the point than what we've seen
        # so far and add it to the index/distance list if positive

        # search the list for the first point with a longer distance
        j = 0
        for nbor in nbors:
            if nbor[1] > d_sq:
                break
            j += 1

        if (j < len(nbors)) or (len(nbors) < k):
            nbors.insert(j, (i, d_sq))
            # too many items in list, get rid of farthest neighbor
            if len(nbors) > k:
                nbors.pop()

    # we now have k nearest neighbors who cast a vote, and the majority
    # wins. we use each class average distance to the target to break any
    # possible ties.

    # a mapping from labels (IDs) to count/distance
    votes = {}
    votes[-1]=[0,0.0]

    for nbor_id, nbor_dist in nbors:
        if nbor_id not in votes:
            votes[nbor_id]=[0,0.0]
        votes[nbor_id][0] += 1
        votes[nbor_id][1] += nbor_dist

        count, dist = votes[nbor_id]

        # check for a possible tie and break with distance
        if (count > votes[winner][0]) or \
                ((count == votes[winner][0]) and (dist < votes[winner][1])):
            winner = nbor_id

    return winner

def track_poses(poses, new_poses):
    # all new blob id's initialized with -1

    # step 1: match new blobs with existing nearest ones
    for i in range(len(poses)):
        winner = find_closest_knn(poses[i], new_poses, 3)

        if winner == -1: # track has died
            poses[i].seen = False # marked for deletion
        else:
            # if winning new blob was labeled winner by another track
            # then compare with this track to see which is closer
            if new_poses[winner].id != -1:
                # find the currently assigned blob
                j = 0
                while j < len(poses):
                    if poses[j].id == new_poses[winner].id:
                        break
                    j += 1

                if j == len(poses): # got to end without finding it
                    new_poses[winner].id = poses[i].id;
                    poses[i] = new_poses[winner]
                else: # found it, compare with current blob
                    dist_old = new_poses[winner].distance_squared( poses[j] )
                    dist_new = new_poses[winner].distance_squared( poses[i] )

                    # if this track is closer, update the id of the pose
                    # otherwise delete this track.. it's dead
                    if dist_new < dist_old: # update
                        new_poses[winner].id = poses[i].id
                        poses[j].seen = False # mark the blob for deletion
                    else: #delete
                        poses[i].seen = False # mark the blob for deletion
            else: # no conflicts, so simply update
                new_poses[winner].id = poses[i].id

    # step 2: pose update
    #
    # update all current tracks
    # remove every track labeled as dead, id = -1
    # find every track that's alive and copy its data from new_poses
    i = 0
    while i < len(poses):
        if not poses[i].is_seen(): # dead
            # erase track
            del poses[i]
            i -= 1 # decrement one since we removed an element
        else: # living
            for j in range(len(new_poses)):
                if poses[i].id == new_poses[j].id:
                    poses[i] = new_poses[j];
        i += 1

    # step 3: add new poses
    # now every new blob should be either labeled with a tracked id or
    # have id of -1. if the id is -1, we need to make a new track.
    global id_counter
    for i in range(len(new_poses)):
        if new_poses[i].id == -1:
            # add new track
            new_poses[i].id = id_counter;
            id_counter += 1;

            poses.append( new_poses[i] )

def track():
    current_poses = []
    i = 1
    for frame in frames:
        sys.stdout.write('\r')
        sys.stdout.write("[%d/%d]" % (i, len(frames)))
        sys.stdout.flush()
        track_poses( current_poses, frame.poses )
        i += 1
    print

def save_frames(frames, filename):
    frame_list = {}
    frame_list['frames'] = []
    for f in frames:
        people = {}
        people["people"] = []
        for p in f.poses:
            people["people"].append({"pose_keypoints" : p.get_flat_keypoints(), \
                "id" : p.id})
        frame_list["frames"].append(people)

    with open(filename, 'w') as outfile:
            json.dump(frame_list, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
            description = 'Experimental tracker for OpenPose keypoints.')
    parser.add_argument('-k', '--keypoints_dir', \
            help = 'OpenPose keypoints directory with .json files')
    parser.add_argument('-o', '--output',
            help = 'Output .json file')
    args = parser.parse_args()
    if args.keypoints_dir:
        load_json_folder(args.keypoints_dir)
        track()
        if args.output:
            save_frames(frames, args.output)
