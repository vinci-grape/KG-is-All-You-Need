import numpy as np
import itertools


class KGCN:
    def __init__(self):
        self.cfg = {
            'triple_path': 'data/triple.npy',
            'threshold': 4.0
        }
        self.triple = np.load(self.cfg['triple_path'], allow_pickle=True).tolist()

    def construct_kg(self):
        """
        Construct knowledge graph
        knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        """
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for k, v in self.triple.items():
            print(k)
            if len(v) > 1:
                for triple in list(itertools.combinations(v, 2)):
                    head = triple[0]
                    relation = k
                    tail = triple[1]
                    if head in kg:
                        kg[head].append((relation, tail))
                    else:
                        kg[head] = [(relation, tail)]
                    if tail in kg:
                        kg[tail].append((relation, head))
                    else:
                        kg[tail] = [(relation, head)]
        print('Done')
        return kg


if __name__ == "__main__":
    kgcn = KGCN()
    kg = kgcn.construct_kg()
    np.save('data/kg.npy', kg)
    print(kg)