import modules.AI as AI


def initialize():
    global model
    model = AI.load_checkpoint().to('cuda')
