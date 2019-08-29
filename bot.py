# Yagyesh's contribution starts here:
#---------------------------- IMPORTING LIBRARIES-----------------------------------
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split

#--------------------------- FLASK ------------------------------

# from flask import Flask, render_template

# app=Flask(__name__)

# @app.route("/")
# def index():
#     return render_template('index.html')
    #tree_use(fit_tree,cols)

# ---------------------- IMPORTING & ANALYSING DATA ----------------------
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()

#----------------------- mapping strings to numbers ------------------------
le = preprocessing.LabelEncoder()
le.fit_transform(y)
y = le.transform(y)


#---------------------- SPLITING & TRANSFORMING DATA-----------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

DTClf  = DecisionTreeClassifier()
fit_tree = DTClf.fit(x_train,y_train)

#---------------------- PRINTING NAME OF DISEASE -------------------

def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease

print("Please reply Yes or No for the following symptoms") 


#----------------------- PREDICTING ON BASIS OF USER RESPONSE---------------------

def tree_use(tree, all_symp):
    tree_ = tree.tree_
    
    feature_name = [
        all_symp[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []

    #print("def tree(,", ".join(all_symp)
    
#___________________________________________________LOGIC___________________
	

    def recurse(node, depth):
#finding if node is leaf or not, if yes then break, else continue.
        if tree_.feature[node] != _tree.TREE_UNDEFINED:	
            threshold = tree_.threshold[node]
#asking for each symptom.
            name = feature_name[node]					
            print(name + " ?")
            ans = input()
            ans = ans.lower()
#If suffering from symptom then modify val.
            if ans == 'yes':							
                val = 1
            else:
    # Else let it be 0.
                val = 0									

#If value less then threshold that means symptom not present, move to next one.
            if  val <= threshold:						
                recurse(tree_.children_left[node], depth + 1)
            else:										
    #If symptom present then move to next possibility i.e. next possible symptom or disease itself.
                symptoms_present.append(name)			
                recurse(tree_.children_right[node], depth + 1)

        else:
#out-------->
#_____________________________________________________________________

            present_disease = print_disease(tree_.value[node])
            print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("symptoms present  " + str(list(symptoms_present)))
            print("symptoms given "  +  str(list(symptoms_given)) )  


            #Small Feature/
            confidence_level = (len(symptoms_present))/len(symptoms_given)
            print("confidence level is " + str(confidence_level))

    recurse(0, 1)

if __name__ == '__main__':
    tree_use(fit_tree,cols)
    # app.run(debug=True)

# Yagyesh's contributio ends here.