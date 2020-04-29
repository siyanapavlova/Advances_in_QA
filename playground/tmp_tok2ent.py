
# prepare the first entity
        entity = entity_stack.pop(0) # tuple: (ID, entity_string)
        entity = (entity[0], entity[1].lower().split()) # tuple: (ID, list(str))
        assert type(entity[1]) is list
        print(f"first entity (ID, mention): {entity[0]} {entity[1]}")  # CLEANUP




""" map node IDs to token indices """
        for i, t in enumerate(self.tokens):
            print(f"\n===== i:{i}, token: {t}")  # CLEANUP

            if t.startswith("##"): # append the wordpiece to the previous token
                accumulated_string += t.strip("#") # add the current token, but without '##'
                acc_count += 1
                print(f"   found a wordpiece. \n   acc._string = {accumulated_string}\n   acc_count = {acc_count}")  # CLEANUP
            else: # nothing special happens.
                accumulated_string = t
                acc_count = 1
                print(f"   no wordpiece.\n   acc._string = {accumulated_string}\n   acc_count = {acc_count}")  # CLEANUP

            if multiword_index != 0 and t != entity[1][multiword_index]: # switch back to out-of-entity mode
                print(f"{t} is not part of the entity!")  # CLEANUP
                multiword_index = 0
                if entity_stack:
                    entity = entity_stack.pop(0) # fetch the next entity
                    entity = (entity[0], entity[1].lower().split())
                    assert type(entity[1]) is list
                    print(f"new entity (ID, tokens): {entity[0]} {entity[1]}")  # CLEANUP

            #TODO 27.04.2020 continue here: not all wordpieces are prefixed with '##'

            if accumulated_string == entity[1][multiword_index]: # entity[1] is a list[str]

                print(f"{accumulated_string} is part of {entity[1]}")  # CLEANUP
                # add all the accumulated token positions to the entity's entry
                if entity[0] not in mapping: # new entry with the ID as key
                    mapping[entity[0]] = [i-acc for acc in range(acc_count)]
                else:
                    mapping[entity[0]].extend([i-acc for acc in range(acc_count)])

                print(f"mapping of {entity[0]}: {mapping[entity[0]]}")  # CLEANUP

                multiword_index += 1  # we may be inside a multi-word entity

                # if we've found as many tokens as needed for the entity
                if len(mapping[entity[0]]) == len(entity[1]) and entity_stack:
                    entity = entity_stack.pop(0)  # fetch the next entity
                    entity = (entity[0], entity[1].lower().split())
                    assert type(entity[1]) is list
                    print(f"new entity (ID, tokens): {entity[0]} {entity[1]}")  # CLEANUP
                    multiword_index = 0