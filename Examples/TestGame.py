import Node as nd
import numpy as np
import MCTS


if __name__ == "__main__":
    RootState = np.array([1.,1.,1.,1.])
    Root = nd.Node(RootState)

    x = MCTS.MCTS(Root)
    x.Run()
