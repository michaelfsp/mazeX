#----------------------------------Useful bits--------------------------------
tosec = lambda h,m,s: 3600*h + 60*m + s

def get_trajectory(timed_centroid, maze, mapping, tolerance):
    tolerance = 30

    nearest_ports = []
    visited_ports = []
    t_axis = []
    last_visited_port = None
    time_visit = 0

    for t, x, y in timed_centroid:
        nearest_port = mapping.find_nearest_port((x,y))
        nearest_ports.append(nearest_port)

        #if mapping.portregions_dic[nearest_port].contains(Point(x,y)):
        if mapping.portregions_dic[nearest_port].distance(Point(x,y)) < tolerance:
            if not last_visited_port:
                visited_ports.append(nearest_port)
                last_visited_port = nearest_port
                t_axis.append(t)
                time_visit = t

            elif nearest_port != last_visited_port:
                if nearest_port in maze.neighbors(last_visited_port):
                    #check if it is connected to the last recognized port,i.e,
                    #  is not a tracking error
                    visited_ports.append(nearest_port)
                    last_visited_port = nearest_port
                    t_axis.append(t)
                    time_visit = t

    return t_axis, visited_ports


