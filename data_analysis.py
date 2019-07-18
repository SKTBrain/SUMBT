# Get class count

import json
import os
import collections
import csv

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='data/woz', type=str)
    args = parser.parse_args()

    ## Parsing ontology
    if args.args.data_dir == "data/woz":
        fp_ontology = open(os.path.join(args.data_dir, "ontology_dstc2_en.json"), "r")
        ontology = json.load(fp_ontology)
        ontology = ontology["informable"]
        del ontology["request"]
        for slot in ontology.keys():
            ontology[slot].append("do not care")
            ontology[slot].append("none")
        fp_ontology.close()

        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        ontology = ontology
        target_slot = list(ontology.keys())
        for i, slot in enumerate(target_slot):
            if slot == "pricerange":
                target_slot[i] = "price range"

    elif args.data_dir == "data/multiwoz":
        fp_ontology = open(os.path.join(args.data_dir, "ontology.json"), "r")
        ontology = json.load(fp_ontology)
        for slot in ontology.keys():
            ontology[slot].append("none")
        fp_ontology.close()

        ontology = collections.OrderedDict(sorted(ontology.items()))
        target_slot = list(ontology.keys())
    else:
        raise NotImplementedError()

    nslot = len(ontology)

    # count data of each slot-value
    count = []
    count_not_none = []
    for slot in ontology.values():
        slot_count = {}
        for val in slot:
            slot_count[val] = [0,0,0] #train, valid, test
        count.append(slot_count)
        count_not_none.append([0,0,0])

    for d, data in enumerate(['train', 'dev', 'test']):
        with open(os.path.join(args.data_dir, "%s.tsv" % data), "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            for l, line in enumerate(reader):
                if (args.data_dir == "data/woz") or (args.data_dir=='data/multiwoz' and l > 0):
                    for s in range(nslot):
                        val = line[4+s]
                        if val == 'dontcare':
                            val = 'do not care'
                        count[s][val][d] += 1
                        if val != 'none':
                            count_not_none[s][d] += 1

    with open(os.path.join(args.data_dir, "train_analysis.txt"), "w", encoding='utf-8') as writer:
        for i, c in enumerate(count):
            writer.write('--- %s --- \n'% target_slot[i])
            for k, v in c.items():
                writer.write('%s\t%d\t%d\t%d\n' % (k, v[0], v[1], v[2]))
        writer.close()

    with open(os.path.join(args.data_dir, "data_not_none_analysis.txt"), "w", encoding='utf-8') as writer:
        domain_data = {}
        for i, slot in enumerate(ontology.keys()):
            domain = slot.split('-')[0]
            v = count_not_none[i]

            if domain not in domain_data:
                domain_data[domain] = [0, 0, 0]

            for j, val in enumerate(v):
                domain_data[domain][j] += val

            writer.write('%s\t%d\t%d\t%d\n' % (slot, v[0], v[1], v[2]))

        writer.write('----- total ----- \n')
        for domain, v in domain_data.items():
            writer.write('%s\t%d\t%d\t%d\n' % (domain, v[0], v[1], v[2]))

        writer.close()

    with open(os.path.join(args.data_dir, "none_ratio.txt"), "w", encoding='utf-8') as writer:
        for i, slot in enumerate(ontology.keys()):
            val = count_not_none[i]
            none = count[i]['none']
            ratio = [ n/(v+n) for v, n in zip(val, none) ]
            writer.write('%s\t:\t%.6e\t%.6e\t%.6e\n' % (slot, ratio[0], ratio[1], ratio[2]))

        writer.close()

    # find common and different slots among domains
    if args.data_dir == "data/multiwoz":
        slot_dict = {}
        for slot in target_slot:
            domain = slot.split('-')[0]
            slot_name = slot.split('-')[1]
            if slot_name not in slot_dict:
                slot_dict[slot_name] = []
            slot_dict[slot_name].append(domain)

        with open(os.path.join(args.data_dir, "domain_slot_analysis.txt"), "w", encoding='utf-8') as writer:
            for slot, domains in slot_dict.items():
                writer.write('%s\t%s\n'%(slot, ' '.join(domains)))
            writer.close()
        print(slot_dict)


