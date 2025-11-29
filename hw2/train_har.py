import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from data_utils import read_data
from feature_utils import create_features
import os

DATA_PATH = os.path.join("..", "data", "WISDM_ar_v1.1_raw.txt")
TIME_PERIODS = 80
STEP_DISTANCE = 40

data_df = read_data(DATA_PATH)
df_train = data_df[data_df["user"] <= 28]
df_test  = data_df[data_df["user"] > 28]

train_segments_df, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
test_segments_df, test_labels   = create_features(df_test , TIME_PERIODS, STEP_DISTANCE)

X_train, X_test, y_train, y_test = train_test_split(train_segments_df, train_labels, test_size=0.25, random_state=42)

clf = GaussianNB()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
cm = confusion_matrix(y_test, preds)

class_names = np.unique(train_labels)
priors = clf.class_prior_
means = clf.theta_
vars_ = clf.sigma_
inv_vars = 1.0 / vars_
dets = np.prod(vars_, axis=1)

with open("../outputs/bayes_har_config.h", "w") as f:
    f.write("#define NUM_CLASSES %d\n" % len(class_names))
    f.write("#define NUM_FEATURES %d\n\n" % X_train.shape[1])
    f.write("static const float CLASS_PRIORS[%d] = {%s};\n\n" %
            (len(class_names), ", ".join([str(x) for x in priors])))

    f.write("static const float MEANS[%d][%d] = {\n" %
            (len(class_names), X_train.shape[1]))
    for i in range(len(class_names)):
        f.write("    {" + ", ".join([str(x) for x in means[i]])
