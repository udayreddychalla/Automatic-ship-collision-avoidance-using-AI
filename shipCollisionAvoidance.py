import sys
import time
import random
import numpy as np
import tkinter as Mazegame
from termcolor import colored
from PIL import ImageTk, Image
from tkinter import ttk, Canvas, Label
import heapq

#This function initializes start and end position of own ship
def startend_postion(n):
    start_x, start_y = map(int,input("Enter start position").split())
    end_x, end_y = map(int,input("Enter end position").split())
    return start_x*n+start_y, end_x*n+end_y


#This function prepares full maze layout
def prepareEnv(n, start, end):
    maze = [[0 for i in range(n)] for j in range(n)]

    maze[start//n][start%n] = 0
    maze[end//n][end%n] = 0
    return maze

#This function makes screen for environment
def make_screen(n):
    if n in range(2,9):
       size = 300
    elif n in range(9,43):
       size = 640
    elif n in range(43, 75):
       size = 750
    elif n in range(75, 200):
       size = 850

    cell_width = int(size/n)
    cell_height = int(size/n)

    screen = Mazegame.Tk()
    screen.title("maritime environment")
    grid = Canvas(screen, width = cell_width*n, height = cell_height*n, highlightthickness=0)
    grid.pack(side="top", fill="both", expand="true")

    rect = {}
    for col in range(n):
        for row in range(n):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            rect[row, col] = grid.create_rectangle(x1,y1,x2,y2, fill="red", tags="rect")
    return grid, rect, screen, cell_width

#This function loads images into specific position from specified path
def load_img(size, path, pos):
    xcod = pos//n
    ycod = pos%n
    load = Image.open(path)
    load = load.resize((size, size), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x = ycod*size, y = xcod*size)
    return img

# This function redraws environment and updates it according to the traversal at 'delay' time interval
def redrawEnv(grid, rect, screen, n, maze, delay, size, pos_list, endShip):
    path1 = "C:/Users/uredd/Desktop/AI_project/ship.jpg"
    grid.itemconfig("rect", fill="green")
    for i in range(n):
        for j in range(n):
            item_id = rect[i,j]
            if maze[i][j] == 0:
                grid.itemconfig(item_id, fill="DeepSkyBlue2")
            elif maze[i][j] == -1:
                grid.itemconfig(item_id, fill="black")
            elif maze[i][j] == 1:
                grid.itemconfig(item_id, fill="salmon")
            elif maze[i][j] == 2:
                grid.itemconfig(item_id, fill="red")
            elif maze[i][j] == 3:
                grid.itemconfig(item_id, fill="magenta")
            elif maze[i][j] == 4:
                grid.itemconfig(item_id, fill="blue")
            elif maze[i][j] == 5:
                grid.itemconfig(item_id, fill="green")
            elif maze[i][j] == 6:
                grid.itemconfig(item_id, fill="cyan")
            elif maze[i][j] == 7:
                grid.itemconfig(item_id, fill="yellow")

    for i in range(n):
        for j in range(n):
            item_id = rect[i,j]
            for row, col in pos_list:
                shipidx = 0
                if row*n+col == i*n+j and shipidx < len(endShip):# and endShip[shipidx] != 1:
                    grid.itemconfig(item_id, fill="black")

                shipidx +=1

    for idx in range(len(pos_list)):
        x,y = pos_list[idx]
        pos = x*n+y
        if idx == len(pos_list)-1:
            load_img(size, path1, pos)


    screen.update_idletasks()
    screen.update()
    time.sleep(delay)
    return

#This function displays button
def button(text, win, window):
    b = ttk.Button(window, text=text, command = win.destroy)
    b.pack()

#This function displayes popup
def popup_win(msg, title, path ,screen):
    popup = Mazegame.Tk()
    popup.wm_title(title)
    label = ttk.Label(popup, text = msg, font=("Times", 20))
    label.pack(side="top", fill="x", pady=50, padx=50)
    button("Close environment", screen, popup)
    button("Close popup", popup, popup)
    popup.mainloop()


#This function returns fuzzy direction of the other ship using their previous and present locations
def getApprochDirection(prevX, prevY, shipX, shipY, othPrevX, othPrevY, otherX, otherY):
    shipDir = direction(prevX, prevY, shipX, shipY)
    otherDir = direction(othPrevX, othPrevY, otherX, otherY)
    if shipDir == 'R':
        if (otherDir == 'RU' or otherDir == 'U' or otherDir == 'LU'):
            return 'right'
        elif (otherDir == 'RD' or otherDir == 'D' or otherDir == 'LD'):
            return 'left'
        else: return 'opposite'

    if shipDir == 'L':
        if (otherDir == 'RU' or otherDir == 'U' or otherDir == 'LU'):
            return 'left'
        elif (otherDir == 'RD' or otherDir == 'D' or otherDir == 'LD'):
            return 'right'
        else: return 'opposite'

    if shipDir == 'U':
        if (otherDir == 'LD' or otherDir == 'L' or otherDir == 'LU'):
            return 'left'
        elif (otherDir == 'RD' or otherDir == 'R' or otherDir == 'RU'):
            return 'right'
        else:
            return 'opposite'

    if shipDir == 'D':
        if (otherDir == 'LD' or otherDir == 'L' or otherDir == 'LU'):
            return 'right'
        elif (otherDir == 'RD' or otherDir == 'R' or otherDir == 'RU'):
            return 'left'
        else:
            return 'opposite'

    if shipDir == 'RD':
        if (otherDir == 'D' or otherDir == 'LD' or otherDir == 'R'):
            return 'right'
        elif (otherDir == 'U' or otherDir == 'RU' or otherDir == 'L'):
            return 'left'
        else:
            return 'opposite'

    if shipDir == 'LU':
        if (otherDir == 'D' or otherDir == 'LD' or otherDir == 'R'):
            return 'left'
        elif (otherDir == 'U' or otherDir == 'RU' or otherDir == 'L'):
            return 'right'
        else:
            return 'opposite'

    if shipDir == 'LD':
        if (otherDir == 'D' or otherDir == 'RD' or otherDir == 'L'):
            return 'left'
        elif (otherDir == 'U' or otherDir == 'LU' or otherDir == 'R'):
            return 'right'
        else:
            return 'opposite'

    if shipDir == 'RU':
        if (otherDir == 'D' or otherDir == 'RD' or otherDir == 'L'):
            return 'right'
        elif (otherDir == 'U' or otherDir == 'LU' or otherDir == 'R'):
            return 'left'
        else:
            return 'opposite'


#This function implements fuzzy logic to estimate costs better way based on the movement of other ship
def fuzzyModule(estTrajectory, prevPos, trajectories, iter, collsionCostData):
    index = iter

    #fuzzy directions --> ['left', 'opposite', 'right']
    dist = [0 for i in range(num_ships)]

    prevX, prevY = prevPos//n, prevPos%n
    startX, startY = int(estTrajectory[0]//n), int(estTrajectory[0]%n)
    for node in estTrajectory:
        x = int(node // n)
        y = int(node % n)
        safeZoneOwn = get_neighbour_list(x, y, x, y)
        safeZoneOwn.append((x, y))
        for ship in range(num_ships):
            if (index < len(trajectories[ship])):
                row, col = trajectories[ship][index]
                pevRow, prevCol = trajectories[ship][index-1]
                if(row,col) == (x,y):

                    directApp = getApprochDirection(prevX, prevY, startX, startY, pevRow, prevCol, row, col)
                    ownDir = direction(prevX, prevY, startX, startY)
                    if dist[ship] > 5: #If other sship is far away
                        continue
                    elif dist[ship] <= 5: #If other ship is near by

                        if directApp == 'left': #If other ship approaching from left
                            if ownDir == 'R' and dist[ship]!=0:
                                collsionCostData[startX][startY+1] += 100/dist[ship]
                                collsionCostData[startX - 1][startY+1] += 100/dist[ship]
                            if ownDir == 'L':
                                collsionCostData[startX][startY-1] += 100/dist[ship]
                                collsionCostData[startX + 1][startY-1] += 100/dist[ship]
                            if ownDir == 'U':
                                collsionCostData[startX - 1][startY] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]
                            if ownDir == 'D':
                                collsionCostData[startX + 1][startY] += 100/dist[ship]
                                collsionCostData[startX + 1][startY+1] += 100/dist[ship]
                            if ownDir == 'RD':
                                collsionCostData[startX][startY+1] += 100/dist[ship]
                                collsionCostData[startX + 1][startY+1] += 100/dist[ship]
                            if ownDir == 'LU':
                                collsionCostData[startX][startY-1] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]
                            if ownDir == 'LD':
                                collsionCostData[startX +1][startY] += 100/dist[ship]
                                collsionCostData[startX +1][startY-1] += 100/dist[ship]
                            if ownDir == 'RU':
                                collsionCostData[startX - 1][startY] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]

                        if directApp == 'right' and dist[ship]!=0: #If other ship approaching from right
                            if ownDir == 'R':
                                collsionCostData[startX][startY+1] += 100/dist[ship]
                                collsionCostData[startX + 1][startY+1] += 100/dist[ship]
                            if ownDir == 'L':
                                collsionCostData[startX][startY-1] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]
                            if ownDir == 'U':
                                collsionCostData[startX - 1][startY] += 100/dist[ship]
                                collsionCostData[startX - 1][startY+1] += 100/dist[ship]
                            if ownDir == 'D':
                                collsionCostData[startX + 1][startY] += 100/dist[ship]
                                collsionCostData[startX + 1][startY-1] += 100/dist[ship]
                            if ownDir == 'LU':
                                collsionCostData[startX][startY+1] += 100/dist[ship]
                                collsionCostData[startX + 1][startY+1] += 100/dist[ship]
                            if ownDir == 'RD':
                                collsionCostData[startX][startY-1] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]
                            if ownDir == 'RU':
                                collsionCostData[startX +1][startY] += 100/dist[ship]
                                collsionCostData[startX +1][startY-1] += 100/dist[ship]
                            if ownDir == 'LD':
                                collsionCostData[startX - 1][startY] += 100/dist[ship]
                                collsionCostData[startX - 1][startY-1] += 100/dist[ship]

                            #If other ship approaches from opposite direction automatically one which is closest to goal node is picked
                dist[ship]+=1
        index+=1
    return collsionCostData


#This function returns direction of movement based on previous and present locations
def direction(x1,y1, x2,y2):

    if x1 == x2:
        if y1 < y2:
            return 'R'
        else:
            return 'L'

    elif y1 == y2:
        if x1 < x2:
            return 'D'
        else:
            return 'U'

    elif x1+1 == x2 and y1+1 == y2:
        return 'RD'

    elif x1-1 == x2 and y1-1 == y2:
        return 'LU'

    elif x1+1 == x2 and y1-1 == y2:
        return 'LD'

    elif x1-1 == x2 and y1+1 ==y2:
        return 'RU'

#This function returns possible next moves based on the direction of its movement
def get_neighbour_list(prevRow, prevCol, row, col):

    if prevRow == row and prevCol == col:
        return [(row - 1, col - 1), (row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1),
                (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)]

    if prevRow == row:
        if prevCol < col: #move right
            return [(row,col+1), (row-1,col+1), (row+1, col+1)]
        else:
            return [(row, col - 1), (row - 1, col - 1), (row + 1, col - 1)]

    elif prevCol == col:
        if prevRow < row: #move down
            return [(row+1, col),(row+1, col-1), (row+1, col+1)]
        else:
            return [(row-1, col),(row-1, col-1), (row-1, col+1)]

    elif prevRow+1 == row and prevCol+1 == col:
        return [(row+1, col+1), (row, col+1), (row+1, col)]

    elif prevRow-1 == row and prevCol-1 == col:
        return [(row - 1, col - 1), (row, col - 1), (row - 1, col)]

    elif prevRow+1 == row and prevCol-1 == col:
        return [(row + 1, col - 1), (row, col - 1), (row +1, col)]

    elif prevRow-1 == row and prevCol+1 ==col:
        return [(row - 1, col + 1), (row, col + 1), (row - 1, col)]


#This function is a combination of output of fuzzy module and A* module
#It returns costs of different positions based on the trajectories of own and other ship
def getCollisionData(n, num_ships, iter, prevPos, estTrajectory, trajectories, step_count):
    collisionCostData = np.zeros((n,n))
    collisionCostData = collisionCostData.astype(int)
    cost = 100
    index = iter

    for node in estTrajectory:
        x = int(node//n)
        y = int(node%n)
        safeZoneOwn = get_neighbour_list(x,y,x,y)
        safeZoneOwn.append((x,y))
        for ship in range(num_ships):
            if(index < len(trajectories[ship])):
                row, col = trajectories[ship][index]
                safeZoneothr = get_neighbour_list(row, col,row, col)
                safeZoneothr.append((row, col))

                #checking for intersection of safe zone of own ship and other chip
                match = set(safeZoneOwn).intersection(safeZoneothr)
                if len(match) > 0:
                    print("chance of safezone overlap detected at : ",x,y)
                    for i,j in match:
                        if i in range(n) and j in range(n) and collisionCostData[i][j] == 0:
                            collisionCostData[i][j] += cost

                #If there is collision between own and other ship
                if x == row and y ==col:
                    print("chance of collision detected at : ", x,y)
                    collisionCostData[row][col]+= 2*cost
        index+=1

    collisionCostData = fuzzyModule(estTrajectory, prevPos, trajectories, iter, collisionCostData)
    return collisionCostData

#This is heuristic function which returns simple manhattan distance
def heuristic(pos, end, collisionCost, n):
    row = pos//n
    col = pos%n
    dist = abs(row - end//n)+abs(col - end%n) + collisionCost[row][col]
    return dist



#This function returns estimated trajectory of ownship using A* with the help of costs computed from fuzzy module and heuristic
def getEstTrajectory(n, collisionCostData, maze, prevPos, start, end):
    pos = start
    parent = np.zeros((n, n))
    visited = np.zeros((n, n))
    state_list = []
    heuristic_cost = heuristic(pos, end, collisionCostData, n)
    prevRow, prevCol = prevPos // n, prevPos % n
    state = [0 + heuristic_cost, 0, int(start//n), int(start%n),prevRow, prevCol]
    heapq.heappush(state_list, state)

    while pos != end:
        if len(state_list) == 0:
            break

        state = heapq.heappop(state_list)

        row = state[2]
        col = state[3]

        if visited[row][col] == 1:
            continue

        pos = row * n + col
        parent_cost = state[1]
        visited[row][col] = 1

        #Here neighbour positions are based on smoothness constraint
        pos_neighbours = get_neighbour_list(state[4], state[5],row, col)
        prevRow, prevCol = row, col
        for (row, col) in pos_neighbours:
            if row in range(n) and col in range(n) and visited[row][col] == 0:
                heurist = heuristic(row * n + col, end, collisionCostData, n)
                source_cost = parent_cost + 1
                state = [source_cost + heurist, source_cost, row, col, prevRow, prevCol]
                parent[row][col] = pos
                heapq.heappush(state_list, state)

    action_list = []

    while True:
        action_list.insert(0, pos)
        row = int(pos // n)
        col = int(pos % n)
        pos = parent[row][col]
        if row == start//n and col == start%n: break


    return action_list


#This function performs ship collision avoidance using estimated trajectory of own ship and-
#trajectories of other ship using A* with fuzzy logic
def shipCollisionAvoidance(n, trajectories, maze, start, end):
    delay = 0.5
    grid, rect, screen, wid = make_screen(n)
    maze[end // n][end % n] = 7
    step_count = np.zeros((num_ships,))
    step_count = step_count.astype(int)
    endShip = np.zeros((num_ships,))
    endShip = endShip.astype(int)
    color_list = [2, 3, 4, 5, 6, 7]
    next_pos = start
    prev_pos = 0

    #This matrix stores collision cost information
    collisionCostData = np.zeros((n,n))
    collisionCostData = collisionCostData.astype(int)
    estTrajectory = getEstTrajectory(n, collisionCostData, maze, prev_pos, next_pos, end)

    iter = 0
    breakEst = 0
    row, col = start//n, start%n
    prevRow, prevCol = 0,0
    breakCounter = 0
    marginCount = 0
    margins = np.zeros((num_ships, 2))
    pos_track = []
    totalCost = 0

    #Iterates until own ship reaches to goal state
    while True:
        print("\ntime step: ", iter)
        next_pos =  row*n + col
        finish = 0
        pos_list = []
        pos_track.append((row,col))

        print("own ship location : (",row,", ",col,")")
        for ship in range(num_ships):
            x,y = trajectories[ship][step_count[ship]]
            pos_list.append((x,y))
            print("ship ",ship+1, "location : ", trajectories[ship][step_count[ship]])
            margin = abs(row-x)+abs(col-y)
            dir1, dir2 = 0,0
            if(step_count[ship]>0):
                x1,y1 = trajectories[ship][step_count[ship]-1]
                dir1 = direction(x1,y1,x,y)
                x1,y1 = prevRow, prevCol
                x2,y2 = row,col
                dir2 = direction(x1, y1, x2, y2)
            if dir1 == dir2:
                margins[ship][marginCount] = margin #margin to decide for break based on direction of movement of other ships
            else:
                margins[ship][marginCount] = 0

        marginCount+=1
        if marginCount == 2:
            marginCount = 0

        pos_list.append((row, col))
        pos = row*n+col
        idx = 0

        #updates colours on maze based on ship movement
        for x, y in pos_list:
            maze[x][y] = color_list[idx]

            if(row!=0 and col!=0 and row == x  and col == y and idx < num_ships and endShip[idx] != 1): #highlights if collision taken place
                list = get_neighbour_list(row, col,row, col)
                for i, j in list:
                    if i in range(n) and j in range(n):
                        maze[i][j] = 1
            idx += 1

        redrawEnv(grid, rect, screen, n, maze, delay, wid, pos_list, endShip)

        for ship in range(num_ships):
            if (step_count[ship] < len(trajectories[ship]) - 1):
                step_count[ship] += 1

        for ship in range(num_ships):
            if (step_count[ship] == len(trajectories[ship]) - 1):
                finish += 1
                endShip[ship] = 1

        if pos == end:finish += 1
        if finish == num_ships+1: break

        #Computing next best move from current position for the own ship
        prevPos = prevRow*n+prevCol
        collisionCostData = getCollisionData(n, num_ships, iter, prevPos, estTrajectory, trajectories, step_count)
        estTrajectory = getEstTrajectory(n, collisionCostData, maze, prevPos, next_pos, end)
        prevRow, prevCol = row, col
        row, col = int(estTrajectory[1] // n), int(estTrajectory[1] % n)
        estTrajectory.pop(0)
        totalCost+=1;

        #Making decision to take break or not
        for ship in range(num_ships):
            if margins[ship][0] == margins[ship][1] and margins[ship][0] != 0 and margins[ship][0] <=3:
                breakEst = 1
                break

        if breakEst == 1:
            breakCounter+=1

        #break will be reflected after 2 grid moves
        if breakCounter == 2:
            row, col = prevRow, prevCol
            prevRow, prevCol = pos_track[-2]
            estTrajectory = getEstTrajectory(n, collisionCostData, maze, prevRow*n+prevCol, row*n+col, end)
            breakEst = 0
            breakCounter=0

        iter += 1

    print("Finished")
    print("\nTotal Cost : ",totalCost)
    popup_win(" ", " ", "./final.png", screen)


#creates random trajectories for other
def get_trajectories(numShips, n):
    trajectories = []
    pos = [0,n-1]
    ship = 0
    while ship != numShips:
        if random.randint(0,1) > 0.5:
            start_x, start_y =  random.choice(pos), random.randint(0,n-1)
        else :
            start_x, start_y = random.randint(0, n-1), random.choice(pos)

        x,y = start_x, start_y
        if(x <2 or y<2):
            continue
        x_dir,y_dir = 0,0
        trajectory = []

        if (x == 0):
            x_dir = 1
        elif (y == 0):
            y_dir = 1
        elif (y == n - 1):
            y_dir = -1
        elif (x == n - 1):
            x_dir = -1

        x1_dir, y1_dir = x_dir, y_dir
        steps = random.choice(range(1, 10))
        while(x in range(n) and y in range(n)):
            trajectory.append((x,y))

            if steps == 0:
                steps = random.choice(range(1, 10))
                if x_dir == 1 or x_dir == -1:
                    y1_dir = random.choice([-1,0,1])
                elif y_dir == 1 or y_dir == -1:
                    x1_dir = random.choice([-1,0,1])

            x = x+x1_dir
            y = y+y1_dir
            steps-=1

        ship+=1
        trajectories.append(trajectory)
    return trajectories

#This function displays collisions possible for own ship by using A* without collision avoidance
def check_trajectories(num_ships, trajectories,n, maze, start, end):
    delay = 0.5
    grid, rect, screen, wid = make_screen(n)
    maze[end // n][end % n] = 7

    step_count = np.zeros((num_ships,))
    step_count = step_count.astype(int)
    endShip = np.zeros((num_ships,))
    endShip = endShip.astype(int)
    color_list = [2, 3, 4, 5, 6, 7]
    next_pos = start
    prev_pos = 0
    collisionCostData = np.zeros((n, n))
    collisionCostData = collisionCostData.astype(int)
    estTrajectory = getEstTrajectory(n, collisionCostData, maze, prev_pos,next_pos, end)
    iter = 0
    row, col = start // n, start % n
    prevRwo, prevCol = start // n, start % n
    while True:

        finish = 0
        pos_list = []

        for ship in range(num_ships):
            pos_list.append(trajectories[ship][step_count[ship]])

        pos_list.append((row, col))
        pos = row * n + col

        idx = 0
        for x, y in pos_list:
            if maze[x][y] != 1:
                maze[x][y] = color_list[idx]

            if(row!=0 and col!=0 and row == x  and col == y and idx < num_ships and endShip[idx] != 1):
                list = get_neighbour_list(row, col,row, col)
                for i,j in list:
                    if i in range(n) and j in range(n):
                        maze[i][j] = 1
            idx += 1
        redrawEnv(grid, rect, screen, n, maze, delay, wid, pos_list, endShip)

        for ship in range(num_ships):
            if (step_count[ship] < len(trajectories[ship]) - 1):
                step_count[ship] += 1

        for ship in range(num_ships):
            if (step_count[ship] == len(trajectories[ship]) - 1):
                finish += 1
                endShip[ship] = 1

        if pos == end: finish += 1

        if finish == num_ships + 1: break

        iter += 1
        prevRwo, prevCol = row, col
        if(iter<len(estTrajectory)):
            row, col = int(estTrajectory[iter] // n), int(estTrajectory[iter] % n)

    popup_win(" ", " ", " ", screen)

def reset_env(n,maze):
    for i in range(n):
        for j in range(n):
            maze[i][j] = 0

if __name__ == "__main__":
    n = 20          #grid environment size nxn(can be modified)
    num_ships = 4   #number of ships(can be modified)
    ship_trajectories = get_trajectories(num_ships, n)
    start, end = 0, (n-1)*n+(n-1) #start and end position of own ship(can be modified)
    maze = prepareEnv(n, start, end)
    check_trajectories(num_ships, ship_trajectories, n, maze, start, end) #To display possible collisions
    reset_env(n, maze)
    shipCollisionAvoidance(n, ship_trajectories, maze, start, end)
