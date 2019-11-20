G = {1:{1:0, 2:1, 3:12},
  2:{2:0, 3:9, 4:3},
  3:{3:0, 5:5},
  4:{3:4, 4:0, 5:13, 6:15},
  5:{5:0, 6:4},
  6:{6:0}}

def Dijkstra(G,v0,v_des,INF=999):
    #distance = [999]*len(G)
    #node_list = [list(G.keys()),distance]
    distances = {vertex: float('infinity') for vertex in G}
    current_pos = v0
    dist = 0
    while(current_pos!=v_des):
        distances.pop(current_pos)
        for neighbor, weight in G[current_pos].items():
            if(neighbor!=current_pos):
                distances[neighbor] = min(distances[neighbor],dist + weight)
        current_pos = min(distances,key=distances.get)
        dist = min(distances.values())
    return dist

print(Dijkstra(G,1,6))