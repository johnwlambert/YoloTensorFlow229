# John Lambert

import tensorflow as tf

def assignSlice( tensor, index, sliceValue):
    #tf.Print(tensor, [tensor],message="Tensor to modify: ")
    originalTensorShape = tensor.get_shape().as_list()
    numElementsInTensor = reduce(lambda x, y: x*y, originalTensorShape)
    tensor = tf.reshape(tensor,shape=[-1])
    before = tf.slice(tensor, [0,],       [index,] )
    after = tf.slice(tensor,  [index+1,], [numElementsInTensor-index-1,])
    valueToInsert = tf.ones( shape=[1], dtype=tf.float32 )
    valueToInsert = tf.mul(valueToInsert,sliceValue)
    tensor = tf.concat(0, [before, valueToInsert, after])
    tensor = tf.reshape(tensor,shape=originalTensorShape)
    return tensor


def unionTF(a,b):
	"""
	Calculates the union of two rectangles, a and b, where each is an array [x,y,w,h]
	"""
    w1 = tf.slice(a,[2,],[1,])
    h1 = tf.slice(a,[3,],[1,])
    w2 = tf.slice(b,[2,],[1,])
    h2 = tf.slice(b,[3,],[1,])
    union = tf.mul(w1,h1)
    union += tf.mul(w2,h2)
    union -= intersectionTF(a,b)
    return union #tf.to_float(union)

def intersectionTF(a,b):
    """
    Calculates the intersection of two rectangles, a and b, where each is an array [x,y,w,h]
    tf.truediv(x, y) # divides x / y elementwise WITH FLOATING PT RESULTS
    """
    two = tf.constant(2,dtype=tf.float32)
    zero = tf.constant(0,dtype=tf.float32)
    x1 = tf.slice(a,[0,],[1,])
    y1 = tf.slice(a,[1,],[1,])
    w1 = tf.slice(a,[2,],[1,])
    h1 = tf.slice(a,[3,],[1,])
    x2 = tf.slice(b,[0,],[1,])
    y2 = tf.slice(b,[1,],[1,])
    w2 = tf.slice(b,[2,],[1,])
    h2 = tf.slice(b,[3,],[1,])

    topA=tf.add(y1,tf.div(h1,two) )
    bottomA=tf.sub(y1,tf.div(h1,two) )
    rightA=tf.add(x1,tf.div(w1,two) )
    leftA=tf.sub(x1,tf.div(w1,two) )

    topB=tf.add(y2,tf.div(h2,two) )
    bottomB=tf.sub(y2,tf.div(h2,two) )
    rightB=tf.add(x2,tf.div(w2,two) )
    leftB=tf.sub(x2,tf.div(w2,two) )

    xOverlap=tf.maximum(zero,tf.minimum(rightA,rightB) - tf.maximum(leftA,leftB) )
    yOverlap=tf.maximum(zero,tf.minimum(topA,topB) - tf.maximum(bottomA,bottomB) )
    return tf.mul(xOverlap, yOverlap)

def iouTF(a,b):
	"""
	Returns the intersection over union of two rectangles, a and b, where each is an array [x,y,w,h]
	"""
    return intersectionTF(a,b)/unionTF(a,b)


def konstantinesCodeButConverted(pred, gt):
    """
    X is (7,7,30)
    gt is (7,7,30)
    assignBool is (7,7,2) -- which prediction has the highest current IOU with the ground truth
    assignVal is (7,7,30) -- the final GT values
    objectIn is (7,7) -- is there an object at all in this grid?
    """
    input_size = pred.get_shape().as_list()
    S1,S2,Tot = input_size
    S = S1
    B = 2
    numVar = 5
    numClass = Tot-(B*numVar)
    #Each of the SxS boxes are boxSize x boxSize, a total image is alwas 448x448
    boxSize = tf.constant(448./S)

    #A tensor where a value of "1" means this box is "responsible" for an object, 0 else
    assignBool = tf.zeros([S,S,B],dtype=tf.float32)
    # A tensor where, if a bounding box is "responsible" for an object, 
    # stores the features of the "ground truth" object it is responsible for
    assignVal = tf.zeros([S, S, B*numVar + numClass])
    zero = tf.constant(0)
    two = tf.constant(2.)
    # Initializes the assignBool and assignVal from "ground" and "X". For each of the possible objects in the
    # ground truth, see which bounding box in X has the highest IOU. Then, assign that bounding box to the true 
    # "ground truth object" For each object in the ground truth we assign that object to the bounding box with 
    # the highest IOU
    for i in range(S):
        for j in range(S):
            for k in range(B):
                # Only check if the ground truth has an object there, otherwise the value ground[i,j, k*5] will = 0
                if tf.not_equal( tf.slice(gt, [i,j, k*5],[1,]) ,tf.constant(0.0)):
                        maxIOUVal = -1
                        maxIOULocation = tf.convert_to_tensor(np.array([-1,-1,-1]))
                        #Now iterating though the pred tensor
                        for l in range(S):
                            for m in range(S):
                                for n in range(B):
                                    pred_bbox = tf.slice(pred,[l,m,(n*5)],[4,])
                                    gt_bbox = tf.slice(gt,[i,j,(k*5)],[4,])
                                    curIOU = iouTF(pred_bbox,gt_bbox)
                                    # If the current IOU is greater than the previous max IOU and a given bounding box has not 
                                    # yet been assigned responsibility for an object, make this bounding box the most likely
                                    # to be "responsible" at this point.
                                    notAssignedYet = tf.slice(assignBool,[l,m,n],[1,])
                                    if tf.logical_and(tf.greater(curIOU,maxIOUVal), tf.equal(notAssignedYet,zero)):
                                        maxIOUVal = curIOU
                                        maxIOULocation = tf.convert_to_tensor(np.array([l,m,n]))
                        a = tf.slice( maxIOULocation, [0,],[1,])
                        b = tf.slice( maxIOULocation, [1,],[1,])
                        c = tf.slice( maxIOULocation, [2,],[1,])
                        indexToAssign = S*a + S*b + c # first two dims have length 7,7
                        assignBool = assignSlice( assignBool, indexToAssign, sliceValue=1) 
       
                        

    # We have another tensor which for each of the SxS boxes has "1" if there is a ground truth "object" in
    # that box and 0 else
    objectIn = tf.zeros(shape=[S,S])
    # We initialize if a given box has a "ground truth" object in it
    for i in range(S):
        for j in range(S):
            # We only consider the box if there isn't already an object in it
            if tf.equal( tf.slice(objectIn,[i,j],[1,]),tf.constant(0)):
                for k in range(B):
                    normalizedX = tf.slice(gt,[i,j,k*5],[1,])
                    if tf.not_equal(normalizedX,zero):
                        indexToAssign = i * S + j
                        objectIn = assignSlice(objectIn,indexToAssign, sliceValue=1)
                        width = tf.slice(gt,[i,j,k*5+2],[1,])
                        height = tf.slice(gt,[i,j,k*5+3],[1,])
                        halfBox = tf.div(boxSize,two)
                        if tf.greater(width,halfBox):
                            boxOut = tf.floordiv(width,halfBox)
                            for l in range(boxOut):
                                WILL BREAK W/ TENSOR
                                # ILLEGAL SYNTAX ABOVE
                                if tf.greater_equal(tf.constant(j-l-1),tf.constant(0)):
                                    indexToAssign = i*S + j-l-1
                                    objectIn = assignSlice(objectIn,indexToAssign, sliceValue=1)
                                if tf.less(tf.constant(j+l+1),tf.constant(S)):
                                	indexToAssign = i*S + j+l+1
                                    objectIn = assignSlice(objectIn,indexToAssign, sliceValue=1)
                        if tf.greater(height,halfBox):
                            boxOut = tf.floordive(height,halfBox)
                            for l in range(boxOut):
                                # WILL BREAK WITH TENSOR
                            if tf.greater_equal(tf.constant(i-l-1),tf.constant(0)):
                            	indexToAssign = (i-l-1)*S + j
                            	objectIn = assignSlice(objectIn,indexToAssign, sliceValue=1)
                            if tf.less(tf.constant(i+l+1),tf.constant(S)):
                            	indexToAssign = (i+l+1)*S + j
                            	objectIn = assignSlice(objectIn,indexToAssign, sliceValue=1)


    # These are the two lambda coefficients of the cost function
    lambdaCoord = tf.constant(5.0)
    lambdaNoObj = tf.constant(0.5)

    # t1-t5 are the five "terms" of the cost function
    t1 = tf.constant(0.)
    t2 = tf.constant(0.)
    t3 = tf.constant(0.)
    t4 = tf.constant(0.)
    t4 = tf.constant(0.)
    t5 = tf.constant(0.)

    # The following calculates the values of terms 1-4 in a single loop through the tensor
    for i in range(S):
        for j in range(S):
            for k in range(B):
	            C1 = tf.slice(assignVal, [i,j,(k*5+4)], [1,])
	            C2 = tf.slice(pred, [i,j,(k*5+4)], [1,] )
	            #Calculating the value of terms 1-3
	            useBox = tf.slice(assignBool, [i,j,k],[1,])
	            if tf.equal( useBox ,tf.constant(1) ):
					#1 values (x1,y1,w1,h1,C1) are the ground truth values
					#2 values (x2,y2,w2,h2,C2) are the estimated truth
					x1 = tf.slice(assignVal,[i,j,k*5])
					x2 = tf.slice(pred,[i,j,(k*5)],[1,])
					y1 = tf.slice(assignVal, [i,j,(k*5+1)], [1,])
					y2 = tf.slice(pred, [i,j,(k*5+1)], [1,])
					w1 = tf.slice( assignVal, [i,j,(k*5+2)], [1,])
					w2 = tf.slice(pred, [i,j,(k*5+2)], [1,])
					h1 = tf.slice(assignVal,[i,j,(k*5+3)], [1,])
					h2 = tf.slice(pred, [i,j,(k*5+3)], [1,])

					t1 += tf.square(x1-x2) + tf.square(y1-y2)
					t2 += tf.square((w1**.5)-(w2**.5)) + tf.square((h1**.5)-(h2**.5))
					t3 += tf.square(C1-C2)
            # Calculating the value of term 4
            else:
                t4 = t4+ (C1-C2)**2

	#The following calculates the values of term 5
	for i in range (0,S):
		for j in range (0,S):
			if objectIn[i,j] ==1:
				for k in range (0,numClass):
					probClass1 = tf.slice(assignVal, [i,j, 10 + k], [1,])
					probClass2 = tf.slice(pred, [i,j,10 + k], [1,] )
					t5 = t5 + tf.square(probClass1-probClass2)

	#We sum up total loss, factoring in lambda values
	totalLoss =  tf.mul(lamdbaCoord,t1) + tf.mul(lamdbaCoord,t2) + t3 + tf.mul(lambdaNoObj,t4) + t5
	return totalLoss


