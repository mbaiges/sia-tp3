import signal
import sys

from exercises import Ej1And, Ej1Xor, Ej2Lineal, Ej2NoLineal, Ej2NoLinealWithTesting, Ej3Xor, Ej3Pair

exercises = {
    'ej1': [
        {
            'name': 'AND',
            'exercise': Ej1And
        },
        {
            'name': 'XOR',
            'exercise': Ej1Xor
        }
    ],
    'ej2': [
        {
            'name': 'Lineal',
            'exercise': Ej2Lineal
        },
        {
            'name': 'No Lineal',
            'exercise': Ej2NoLineal
        },
        {
            'name': 'No Lineal With Testing',
            'exercise': Ej2NoLinealWithTesting
        }
    ],
    'ej3': [
        {
            'name': 'XOR',
            'exercise': Ej3Xor
        },
        {
            'name': 'Pair',
            'exercise': Ej3Pair
        },
        
    ],
}

def error():
	print('Not a valid entry. Please try again')

def sigint_handler(sig, frame):
    print('\nExiting')
    sys.exit(0)

if __name__ == "__main__":
    # sets SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)

    # prompt for Exercise selection

    ex_selected = False

    while not ex_selected or not (ex_chosen >= 1 and ex_chosen <= len(exercises.keys())):
        if (ex_selected):
            error()
        else:
            ex_selected = True
        print("All exercises:")
        ex_idx = 0
        for exercise in exercises.keys():
            ex_idx += 1
            print("%s - %s" % (f'{ex_idx:03}', exercise) )

        try:
            ex_chosen = int(input("Please select an exercise: "))
        except ValueError:
            ex_chosen = -1

    # determine exercise
    ex_chosen -= 1

    sub_exercises = list(exercises.values())[ex_chosen]

    sub_chosen = 0

    if len(sub_exercises) > 1:
        # prompt for Exercise selection

        sub_selected = False

        while not sub_selected or not (sub_chosen >= 1 and sub_chosen <= len(sub_exercises)):
            if (sub_selected):
                error()
            else:
                sub_selected = True
            print("All sub exercises:")
            sub_idx = 0
            for sub in sub_exercises:
                sub_idx += 1
                print("%s - %s" % (f'{sub_idx:03}', sub['name']) )

            try:
                sub_chosen = int(input("Please select a sub exercise: "))
            except ValueError:
                sub_chosen = -1

        # determine sub exercise
        sub_chosen -= 1
        
    sub_exercise = sub_exercises[sub_chosen]

    print(sub_exercise['name'])

    ej = sub_exercise['exercise']()

    actions = [
        {
            'name': 'Train and Test',
            'func': ej.train_and_test
        },
        {
            'name': 'Predict',
            'func': ej.predict
        }
    ]

    action_selected = False

    while not action_selected or not (action_chosen >= 1 and action_chosen <= len(actions)):
        if (action_selected):
            error()
        else:
            action_selected = True
        print("All actions:")
        action_idx = 0
        for action in actions:
            action_idx += 1
            print("%s - %s" % (f'{action_idx:03}', action['name']) )

        try:
            action_chosen = int(input("Please select an action: "))
        except ValueError:
            action_chosen = -1

    # determine exercise
    action_chosen -= 1

    action = actions[action_chosen]['func']

    action()