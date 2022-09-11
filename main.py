import numpy as np
from tensorflow.keras.optimizers import Adam

for source_user in ['S1', 'S2', 'S3']:
    for target_user in ['S1', 'S2', 'S3']:
        source_user = str(source_user)
        target_user = str(target_user)
        if source_user == target_user:
            continue
        # problem mappings: 8_3  4_7   4_2  3_8

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # get features read files
        # source_user = '3'  # 2, 3, 4, 5, 7, 8
        # target_user = '8'
        Sampling_frequency = 30  # HZ
        Num_Seconds = 0.3
        Window_Overlap_Rate = 0.5
        DATASET_NAME = 'OPPT'
        activities_required = ['Stand', 'Walk', 'Sit', 'Lie']
        n_state = 4
        n_activities = 4
        n_feature = 76
        Cov_Type = 'diag'
        # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
        # /////////////////
        with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            all_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_features.npy', 'rb') as f:
            all_target_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
            all_source_labels = np.load(f)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
            all_target_labels = np.load(f)
        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        from tslearn.shapelets import LearningShapelets

        model = LearningShapelets(n_shapelets_per_size={10: 2},
                                  weight_regularizer=0.0001,
                                  optimizer=Adam(lr=0.01),
                                  max_iter=300,
                                  verbose=0,
                                  scale=False,
                                  random_state=42)
        model.fit(all_source_bags, all_source_labels)
        train_distances = model.transform(all_source_bags)
        s_accuracy = model.score(all_source_bags, all_source_labels)
        test_distances = model.transform(all_target_bags)
        t_accuracy = model.score(all_target_bags, all_target_labels)
        shapelets = model.shapelets_as_time_series_

        '''
        all_source_bags = all_source_bags.astype(float)

        gmm = mixture.GaussianMixture(n_components=4, covariance_type='diag')
        gmm.fit(all_source_bags)
        gmm_labels = gmm.predict(all_source_bags)

        
        n = len(all_source_bags)
        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
        # loss matrix
        M = ot.dist(all_source_bags)
        plt.figure()
        plt.imshow(M, interpolation='nearest')
        plt.title('Cost matrix M')
        plt.show()


        aa = all_source_bags[:, 0]
        Model_a_act_s = HMM.HMM_with_no_restricts(all_source_bags, source_user, 4, 'diag', 'all')
        hmm_labels = Model_a_act_s.predict(all_source_bags)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(all_source_bags)
        kmeans_labels = kmeans.labels_
        true_labels = all_source_labels
        print()
        '''

