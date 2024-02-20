import numpy as np

def select_from_scores(all_scores, selection="lex", elitism=False, epsilon=False, num_to_select=None):
        if num_to_select is None:
            num_to_select = all_scores.shape[0]
        if num_to_select == 1 and elitism:
            print("WARNING: elitism is not compatible with num_to_select=1. This will just return the best individual")
        selected = []
        if elitism:
            print(f" MAX SCORE: {np.max(np.sum(all_scores, axis=1))}")
            print(f"index of MAX score: {np.argmax(np.sum(all_scores, axis=1))}")
            selected.append(np.argmax(np.sum(all_scores, axis=1))) #elitism

        if selection == "lex":
            if epsilon:
                # do lexicase selection w/ epsilon
                x_median = np.median(all_scores, axis=1)
                # Calculate absolute deviation from median
                dev = abs(all_scores - x_median[:,None])
                mad = np.median(dev, axis=0)
            else:
                mad = np.zeros(all_scores.shape[1])

            for itr in range(
                #only start at 0 if not elitism
                int(elitism), num_to_select
            ):  # , desc='selected', file=sys.stdout):
                num_features = all_scores.shape[1] #8
                features = np.arange(num_features)
                np.random.shuffle(features)
                pool = np.ones(all_scores.shape[0], dtype=bool)  # logical array if selected
                depth = 0
                while len(features) != 0 and np.sum(pool) != 1:  # while we still have cases to use
                    depth += 1
                    feature = features[0]
                    features = features[1:]
                    
                    best = np.max(all_scores[pool, feature])
                    old_pool = pool
                    # print(f"old pool: {old_pool}")

                    # filter selected pop with this feature. If it filters everyone, skip
                    pool = np.logical_and(
                        pool, all_scores[:, feature] >= best - mad[feature],
                    )  
                #print(f"depth: {depth}")
                selected.append(np.argmax(pool))


        elif selection == "tourn":
            agg_scores = np.sum(all_scores, axis=1)
            
            for itr in range(int(elitism), all_scores.shape[0]):
                #tournament selection of one individual
                parents = np.random.choice(np.arange(all_scores.shape[0]), (all_scores.shape[0] // 10,))
                best_parent = np.argmax(agg_scores[parents])
                selected.append(parents[best_parent])
        elif selection == "rand":
            for itr in range(int(elitism), all_scores.shape[0]):
                selected.append(np.random.randint(0, all_scores.shape[0]))
        elif selection == "fps":
            #fitness proportional selection
            agg_scores = np.sum(all_scores, axis=1)
            probs = agg_scores / np.sum(agg_scores)
            selected = np.random.choice(np.arange(all_scores.shape[0]), (all_scores.shape[0],), p=probs)
        else:
            raise Exception("Invalid selection method")
        
        return selected