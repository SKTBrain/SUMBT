import json

source_files = ["woz_train_en.json", "woz_validate_en.json", "woz_test_en.json"]
target_files = ["../train.tsv", "../dev.tsv", "../test.tsv"]

target_slots = ["area", "food", "price range"]

fp_ont = open("ontology_dstc2_en.json", "r")
ontology = json.load(fp_ont)
ontology = ontology["informable"]
for slot in ontology.keys():
    ontology[slot].append("dontcare")
fp_ont.close()

for idx, src in enumerate(source_files):
    trg = target_files[idx]

    fp_src = open(src, "r")
    fp_trg = open(trg, "w")

    data = json.load(fp_src)

    for dialogue in data:
        dialogue_idx = dialogue["dialogue_idx"]
        for turn in dialogue["dialogue"]:
            turn_idx = turn["turn_idx"]
            user_utterance = turn["transcript"]
            system_response = turn["system_transcript"]
            belief_state = turn["belief_state"]

            # initialize turn label and belief state to "none"
            belief_st = {}
            for ts in target_slots:
                belief_st[ts] = "none"

            # extract slot values in belief state
            for bs in belief_state:
                for slots in bs["slots"]:
                    if slots[0] in belief_st:
                        if slots[1] == "center": slots[1] = "centre"
                        if slots[1] == "east side": slots[1] = "east"
                        if slots[1] == "corsican": slots[1] = "corsica"
                        if slots[1] == " expensive": slots[1] = "expensive"

                        assert(belief_st[slots[0]] == "none" or belief_st[slots[0]] == slots[1])
                        assert(slots[1] in ontology[slots[0]])
                        belief_st[slots[0]] = slots[1]

            fp_trg.write(str(dialogue_idx))                 # 0: dialogue index
            fp_trg.write("\t" + str(turn_idx))              # 1: turn index
            fp_trg.write("\t" + str(user_utterance.replace("\t", " ")))        # 2: user utterance
            fp_trg.write("\t" + str(system_response.replace("\t", " ")))       # 3: system response

            for slot in sorted(belief_st.keys()):
                fp_trg.write("\t" + str(belief_st[slot]))   # 4-6: belief state

            fp_trg.write("\n")
            fp_trg.flush()

    fp_src.close()
    fp_trg.close()
