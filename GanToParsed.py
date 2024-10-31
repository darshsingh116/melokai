import math

def convertHitobjects(input_data, beatlen, starttime):
    
    temp = []
    output = []
    for i in input_data: # i = [time,x,y,type(0,1,2,3,4,5,6,7)]

        if i[3] == 1:
            output.append(processCircle(i, beatlen, starttime))
        
        elif i[3]==2:
            temp = []
            temp.append(i)

        elif i[3]==3:
            temp.append(i)

        elif i[3]==4:
            temp.append(i)
            output.append(processSlider(temp, beatlen, starttime))
        
        elif i[3]==5:
            temp = []
            temp.append(i)

        elif i[3]==6:
            temp.append(i)

        elif i[3]==7:
            temp.append(i)
            print("process spinner")
            output.append([])

    return output


def processCircle(hitobject, beatlen,starttime): #hitobject is 1D array
    return [hitobject[1], hitobject[2], hitobject[0], hitobject[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, hitobject[3], hitobject[3], 100, 0, 0, 0, 0, 0, 0, starttime, beatlen, 0, 0, 0, 0, 0, 0]

            
def processSlider(temp, beatlen, starttime): #only output L or P type
    templen = len(temp)
    if templen == 2:
        return processLinearSlider(temp, beatlen, starttime)
    elif templen > 2:
        x1,x2,x3 = temp[0][1],temp[1][1],temp[2][1]
        y1,y2,y3 = temp[0][2],temp[1][2],temp[2][2]
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if area < 125:
            return processLinearSlider(temp, beatlen, starttime)
        elif templen == 3:
            return processPerfectCircleSlider(temp, beatlen, starttime)
        else:
            print("BAZIER FOUND")
            return processLinearSlider(temp, beatlen, starttime)
    else:
        ValueError("SLIDER WITH 1 DATA POINT")


def processLinearSlider(temp, beatlen, starttime, SliderMultiplier):
    output = []
    x,y = temp[0][1],temp[0][2]
    repeat = 0
    unique = 0
    # a b c b a b c b a like repeat
    for t in temp:
        if x==t[1] and y==t[2]:
            repeat += 1

        if repeat <= 2:
            unique += 1

    if repeat != 1:
        unique = (unique//2) + 1
    
    #now we know how many unique

    if not (x==temp[-1][1] and y==temp[-1][2]):
        repeat += 1

    #now we know how many unique also
    totalLen = 0
    for i in range(unique-1):
        totalLen = distance(temp[i][1],temp[i][2],temp[i+1][1],temp[i+1][2])

    totalTime =  temp[unique-1][0] - temp[0][0]

    svm = (totalLen * beatlen)/(SliderMultiplier * 100 (totalTime))
    timingPointSliderVelocityMultiplier = (100/svm)

    for i in range(unique):
            output.append([temp[i][1], temp[i][2], temp[i][0], 0, 0, 2, temp[i][1], temp[i][2], repeat, (totalLen/unique), 0, 0, 0, 0, 0, 0, (temp[i][1]-2), 0, temp[0][3], temp[0][3], timingPointSliderVelocityMultiplier, 0, 0, 0, 0, 0, 0, starttime, beatlen, 0, 0, 0, 0, 0, 0])

    return output


    

def processPerfectCircleSlider(temp, beatlen, starttime, SliderMultiplier):
    output = []
    x,y = temp[0][1],temp[0][2]
    repeat = 0
    unique = 0
    # a b c b a b c b a like repeat
    for t in temp:
        if x==t[1] and y==t[2]:
            repeat += 1

        if repeat <= 2:
            unique += 1

    if repeat != 1:
        unique = (unique//2) + 1
    
    #now we know how many unique

    if not (x==temp[-1][1] and y==temp[-1][2]):
        repeat += 1

    #now we know how many unique also
    totalLen = circle_arc_length(temp[0][1],temp[0][2],temp[1][1],temp[1][2],temp[2][1],temp[2][2])
    totalTime =  temp[2][0] - temp[0][0]

    svm = (totalLen * beatlen)/(SliderMultiplier * 100 (totalTime))
    timingPointSliderVelocityMultiplier = (100/svm)

    for i in range(unique):
            output.append([temp[i][1], temp[i][2], temp[i][0], 0, 0, 3, temp[i][1], temp[i][2], repeat, (totalLen/3), 0, 0, 0, 0, 0, 0, (temp[i][1]-2), 0, temp[0][3], temp[0][3], timingPointSliderVelocityMultiplier, 0, 0, 0, 0, 0, 0, starttime, beatlen, 0, 0, 0, 0, 0, 0])

    return output

def distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def circle_arc_length(x1, y1, x2, y2, x3, y3):
    # Step 1: Calculate the center and radius of the circle
    def circle_center(x1, y1, x2, y2, x3, y3):
        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
        C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
        D = (x1**2 + y1**2) * (x3 * y2 - x2 * y3) + (x2**2 + y2**2) * (x1 * y3 - x3 * y1) + (x3**2 + y3**2) * (x2 * y1 - x1 * y2)
        
        h = -B / (2 * A)
        k = -C / (2 * A)
        r = math.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2))
        
        return h, k, r
    
    # Step 2: Calculate the angle subtended by the arc
    def angle_between_points(h, k, x1, y1, x2, y2, x3, y3):
        
        
        # Calculate vectors from the center to each point
        a = distance(h, k, x1, y1)
        b = distance(h, k, x3, y3)
        c = distance(x1, y1, x3, y3)
        
        # Use cosine rule to calculate the angle between points
        theta = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
        
        return theta
    
    # Step 3: Calculate arc length
    h, k, r = circle_center(x1, y1, x2, y2, x3, y3)
    theta = angle_between_points(h, k, x1, y1, x2, y2, x3, y3)
    arc_length = r * theta
    
    return arc_length


# def processSpinner(temp, beatlen, starttime, SliderMultiplier):
    