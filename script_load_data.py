import os
import glob

import mazex as mx


def maze_for_session(mouse, year, month, day):
    """Returns the appropriate maze object.
    """
    fully_connected_maze = mx.RandomButConnectedSquareMaze(6, 6*5*2)

    original_maze = mx.SquareMaze(6)
    h_connections = [[(0,0), (1,0)], [(1,0), (2,0)], [(2,0), (3,0)], [(3,0), (4,0)],
            [(3,1), (4,1)],
            [(0,2), (1,2)], [(2,2), (3,2)], [(3,2), (4,2)], [(4,2), (5,2)],
            [(1,3), (2,3)], [(3,3), (4,3)],
            [(0,4), (1,4)], [(2,4), (3,4)], [(3,4), (4,4)], [(4,4), (5,4)],
            [(0,5), (1,5)], [(1,5), (2,5)], [(2,5), (3,5)], [(4,5), (5,5)]]

    v_connections = [[(0,0), (0,1)], [(0,1), (0,2)], [(0,3), (0,4)], [(0,4), (0,5)],
            [(1,1), (1,2)], [(1,2), (1,3)], [(1,3), (1,4)],
            [(2,0), (2,1)], [(2,2), (2,3)], [(2,3), (2,4)], [(2,4), (2,5)],
            [(3,0), (3,1)], [(3,3), (3,4)],
            [(4,1), (4,2)], [(4,2), (4,3)], [(4,4), (4,5)],
            [(5,0), (5,1)], [(5,1), (5,2)], [(5,2), (5,3)], [(5,3), (5,4)]]

    original_maze.add_edges_from(h_connections)
    original_maze.add_edges_from(v_connections)

    maze1_str = '''
._._._._._.
| . |_._|_|
|_| | |_|_.
|_| ._. ._.
|_| . | |_|
|_._|_|_|_|
'''
    maze2_mod_str = '''
._._._._. .
._._| ._|_|
|_. | | |_.
| . |_|_._|
| |_. ._|_.
|_|_|_|_. |
'''

    maze2_str = '''
._._._._. .
|_._| ._|_|
|_. | | |_.
| . |_|_._|
| |_. ._._|
|_|_._|_._|
'''
    maze1 = mx.mazes.maze_from_ASCII(maze1_str)
    maze2_mod = mx.mazes.maze_from_ASCII(maze2_mod_str)
    maze2 = mx.mazes.maze_from_ASCII(maze2_str)

    if int(year) == 2019:
        if int(month) == 8:
            if int(day) >= 8 and int(day) <= 17:
                # fully connected maze
                maze = fully_connected_maze
            elif int(day) >= 19 and int(day) <= 28:
                maze = original_maze

            elif int(day) >= 30 and int(day) <= 31:
                maze = maze1

        elif int(month) == 9:
            if int(day) >= 1 and int(day) <= 10:
                maze = maze1

            elif int(day) == 12:
                if mouse == '10I' or mouse == '10II':
                    maze = maze2_mod

                else:
                    maze = maze2

            elif int(day) >= 13 and int(day) <= 21:
                maze = maze2

    return maze

maze_names = ['fully','original','maze1','maze2']


##
#------------------------------------ Parameters -----------------------------
mice = ['10I', '10II', '10III', '11I', '11II', '12I', '12II', '12III']

experiment_days = {'fully': [
    ['2019', '08', '08'],
    ['2019', '08', '09'],
    ['2019', '08', '10'],
    ['2019', '08', '11'],
    ['2019', '08', '12'],
    ['2019', '08', '13'],
    ['2019', '08', '14'],
    ['2019', '08', '15'],
    ['2019', '08', '16'],
    ['2019', '08', '17']
],
                   'original': [
    ['2019', '08', '19'],
    ['2019', '08', '20'],
    ['2019', '08', '21'],
    ['2019', '08', '22'],
    ['2019', '08', '23'],
    ['2019', '08', '24'],
    ['2019', '08', '25'],
    ['2019', '08', '26'],
    ['2019', '08', '27'],
    ['2019', '08', '28']
],
                   'maze1': [
    ['2019', '08', '30'],
    ['2019', '08', '31'],
    ['2019', '09', '01'],
    ['2019', '09', '02'],
    ['2019', '09', '03'],
    ['2019', '09', '04'],
    ['2019', '09', '05'],
    ['2019', '09', '06'],
    ['2019', '09', '07'],
    ['2019', '09', '08'],
    ['2019', '09', '09'],
    ['2019', '09', '10']
],
                   'maze2': [
    ['2019', '09', '12'],
    ['2019', '09', '13'],
    ['2019', '09', '14'],
    ['2019', '09', '15'],
    ['2019', '09', '16'],
    ['2019', '09', '17'],
    ['2019', '09', '18'],
    ['2019', '09', '19'],
    ['2019', '09', '20'],
    ['2019', '09', '21']
]
}

fully_connected_maze = mx.RandomButConnectedSquareMaze(6, 6*5*2)
maze = fully_connected_maze

skip_first_and_last_trials = True

fps = 60


background_file = '/home/michael/code/mousemaze/figures/stills/background_2019.svg'
background_PNG = '/home/michael/code/mousemaze/figures/stills/still_2019.png'

arduino_data_folder = '/mnt/dataz/michael/git-annex/client/data/data/interim/bigmaze/2019/pokestr_fixed'
dlc_data_folder = '/mnt/dataz/michael/git-annex/client/data/data/processed/bigmaze/2019/deeplabcut/resnet50-it103'
bonsai_metadata_folder = '/mnt/dataz/michael/git-annex/client/data/data/interim/bigmaze/2019/video_metadata'

date = ['2019', '09', '19']
mouse = '10II'

ddd = '{}{}{}'.format(*date)
arduino_file_paths = glob.glob(os.path.join(arduino_data_folder, mouse+'_'+ddd+'**.txt'), recursive=True)
if len(arduino_file_paths) != 1:
    raise Exception('There are more or less than exactly one arduino data file for mouse {} and date {}'.format(mouse, date))
else:
    arduino_file_path = arduino_file_paths[0]

arduino_data = mx.data.arduino.ArduinoDataFile(arduino_file_path)

d_d_d = '{}-{}-{}'.format(*date)
bonsai_metadata_file_paths = glob.glob(os.path.join(bonsai_metadata_folder, 'metadata_'+mouse+'_'+d_d_d+'**.csv'), recursive=True)
dlc_coords_file_paths = glob.glob(os.path.join(dlc_data_folder, mouse+'_'+d_d_d+'**.h5'), recursive=True)

if len(bonsai_metadata_file_paths) != 1:
    raise Exception('There are more or less than exactly one bonsai metadata file for mouse {} and date {}'.format(mouse, date))
else:
    bonsai_metadata_file_path = bonsai_metadata_file_paths[0]

if len(dlc_coords_file_paths) != 1:
    raise Exception('There are more or less than exactly one Deeplabcut results file for mouse {} and date {}'.format(mouse, date))
else:
    dlc_coords_file_path = dlc_coords_file_paths[0]


maze = maze_for_session(mouse, *date)

ses = mx.data.load.load_session(background_file, background_PNG, arduino_file_path, bonsai_metadata_file_path, dlc_coords_file_path, maze, date = date, mouse = mouse)

steps_df = mx.data.load.load_steps_from_single_session(ses)

date2 = ['2019', '09', '20']

ses2 = mx.data.load.load_session(background_file, background_PNG, arduino_file_path, bonsai_metadata_file_path, dlc_coords_file_path, maze, date = date2, mouse = mouse)

steps_sessions_df = mx.data.load.load_steps_from_several_sessions([ses, ses2])
