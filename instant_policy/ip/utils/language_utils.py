import random


TASK_TEMPLATES = {
    'lift_lid': [
        'Lift the lid off the saucepan.',
        'Remove the lid from the pot.',
        'Pick up the lid and move it up.',
    ],
    'phone_on_base': [
        'Place the phone on its base.',
        'Put the phone onto the base station.',
        'Set the phone down on the charging base.',
    ],
    'open_box': [
        'Open the box.',
        'Lift the box lid.',
        'Open the container.',
    ],
    'slide_block': [
        'Slide the block to the target area.',
        'Move the block onto the target.',
        'Push the block to the goal region.',
    ],
    'close_box': [
        'Close the box.',
        'Shut the box lid.',
        'Close the container.',
    ],
    'basketball': [
        'Put the ball through the hoop.',
        'Score by placing the ball in the hoop.',
        'Drop the ball into the basketball hoop.',
    ],
    'buzz': [
        'Guide the ring along the wire without touching.',
        'Move the ring along the wire carefully.',
        'Slide the ring along the buzz wire.',
    ],
    'close_microwave': [
        'Close the microwave door.',
        'Shut the microwave.',
        'Push the microwave door closed.',
    ],
    'plate_out': [
        'Take the plate off the dish rack.',
        'Remove the plate from the rack.',
        'Lift the plate out of the rack.',
    ],
    'toilet_seat_down': [
        'Lower the toilet seat.',
        'Put the toilet seat down.',
        'Close the toilet seat.',
    ],
    'toilet_seat_up': [
        'Raise the toilet seat.',
        'Lift the toilet seat up.',
        'Open the toilet seat.',
    ],
    'toilet_roll_off': [
        'Take the toilet roll off the stand.',
        'Remove the toilet roll from the holder.',
        'Lift the roll off the stand.',
    ],
    'open_microwave': [
        'Open the microwave door.',
        'Pull the microwave door open.',
        'Open the microwave.',
    ],
    'lamp_on': [
        'Switch the lamp on.',
        'Turn on the lamp.',
        'Activate the lamp.',
    ],
    'umbrella_out': [
        'Take the umbrella out of the stand.',
        'Remove the umbrella from the holder.',
        'Pull the umbrella out.',
    ],
    'push_button': [
        'Press the button.',
        'Push the button down.',
        'Activate the button.',
    ],
    'put_rubbish': [
        'Put the rubbish in the bin.',
        'Throw the trash into the bin.',
        'Place the rubbish into the bin.',
    ],
}


def get_language_description(task_name, rng=None):
    templates = TASK_TEMPLATES.get(task_name, [task_name.replace('_', ' ')])
    if rng is None:
        return random.choice(templates)
    return templates[int(rng.integers(0, len(templates)))]


def encode_texts(texts, model_name='all-mpnet-base-v2', device='cpu'):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'sentence-transformers is required to encode language. '
            'Install it or precompute lang_emb and skip encoding.'
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, convert_to_tensor=True)
