import os

models = [
    "task=bert-gru",
    "task=bert-lstm",
]

target_slots = ["area", "food", "pricerange"]

for model in models:
    results = {}

    for target_slot in target_slots:
        results[target_slot] = []
        model_dir = "models/%s-slot=%s" % (model, target_slot)
        try:
            with open(os.path.join(model_dir, "eval_corr_results.txt"), "r") as fp:
                for line in fp:
                    results[target_slot].append(int(line.strip()))
        except FileNotFoundError:
            break

    nb_examples = len(results[target_slot])
    nb_all_corr = 0
    for i in range(nb_examples):
        all_corr = True
        for target_slot in target_slots:
            if results[target_slot][i] == 0:
                all_corr = False
        if all_corr:
            nb_all_corr += 1

    if nb_examples > 0:
        print("Joint accuracy (%s-trainable=%s-distance=%s): %f (%d/%d)" % (model,
                                                   nb_all_corr/nb_examples, nb_all_corr, nb_examples))
    else:
        print("Joint accuracy (%s-trainable=%s-distance=%s): nb_example=0" % model)
