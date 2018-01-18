import numpy as np 
import cv2

cam  = cv2.VideoCapture(0)

face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


font  = cv2.FONT_HERSHEY_SIMPLEX

c=0

names={}

file=open("names.txt","r")

for line in file:
    names[c]=line[0:-1]
    c+=1

print(names)
data=np.array([])
for key,value in names.items():
    print(value)
    value=value+".npy"
    d=np.load(value).reshape(50,50*50*3)
    if(key==0):
        data=d
    else:
        data=np.vstack((data,d))





score=0.82
labels  = np.zeros((c*50,1))
a=0
for i in range(c):
    labels[a:a+50]=float(i)
    a+=50



print (data.shape ,  labels.shape)#gi+ves training data and labels

def convolution(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, padding='SAME', strides=[1,strides, strides,1])
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpooling(x, k = 2):
    return tf.nn.max_pool(x, padding='SAME', ksize=[1, k, k, 1], strides=[1,k, k, 1])
    
def cnn_result(x, weights, biases) :
    x = tf.reshape(x, shape=[-1, 48, 48, 1])
    conv1 = conv(x, weights['wc1'], biases['bc1'])
    conv1 = maxpooling(conv1, k = 2)
    
    conv2 = conv(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpooling(conv2, k = 2)
    
    hidden_input = tf.reshape(conv2, [-1, 12 * 12 * 64])
    hidden_output_before_relu = tf.add(tf.matmul(hidden_input, weights['whl']), biases['bhl'])
    hidden_output = tf.nn.relu(hidden_output_before_relu)
    
    out = tf.add(tf.matmul(hidden_output, weights['out']), biases['outB'])
    return out
def trainning(x,weight,biases):
    pred = cnn_result(x, weights, biases)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
def calscore():
    count = 0
    var=0
    init = tf.global_variables_initializer()
    batch_size = 100
    sess = tf.Session()
    sess.run(init)
    t0=time()
    for i in range(15):
        num_batches = 269
        total_cost = 0
        total = 0
        x_train,y_train=split()
        var=random.uniform(1.5,3.5)
        count+=var
        start_batch=0
        for j in range(num_batches):
            batch_x, batch_y = next_batch(batch_size,x_train,y_train)
            _,c,cp = sess.run([optimizer,cost,correct_prediction], feed_dict={x: batch_x, y : batch_y})
            cp.sort()
            index=0
            while(1):
                if(cp[index]):
                    break
                else:
                    index+=1
            
            total +=(len(cp)-index)/len(cp)
            total_cost += c
        cs=score(total,count)
def pickle():
    import pickle
    save_svc_grid_search=open("cnn.pickle","wb")
    pickle.dump(clf,save_svc_grid_search)
    save_svc_grid_search.close()

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())
                       
def knn(x , train , targets  , k = 5 ):  
    m = train.shape[0]
    dist = []
    for ix  in range(m):
        # storing dist from each point in dist list
        dist.append(distance(x ,train[ix]))
        pass
    
    dist = np.asarray(dist)
    indx  = np.argsort(dist)
    
    sorted_labels  = labels[indx][:k ]
    
    counts  = np.unique(sorted_labels , return_counts=True)
    
    return counts[0][np.argmax(counts[1])]

while True:

    ret , frame = cam.read()

    if ret == True :
        #convert to grayscale and get the faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces  = face_cas.detectMultiScale(gray , 1.3 , 5 )

        for (x , y , w , h) in faces:
            face_component = frame[y:y+h , x:x+w , :]
            fc = cv2.resize(face_component, (50,50))

            
            lab = knn(fc.flatten() , data , labels )


            
            text = names[int(lab)]

            #text generated appears on frame
            cv2.putText(frame , text , (x,y) , font , 1,(255,255,0) , 2)

            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,0,255), 2)

        cv2.imshow('face recognition' , frame)

        if cv2.waitKey(1)  ==  27:
            break  

    else :
        print('error\n')

cv2.destroyAllWindows()
