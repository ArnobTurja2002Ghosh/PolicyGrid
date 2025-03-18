# PolicyGrid
Consider a robot operating in the following environment which has 9 free spaces labelled purple:

<img src= "docs/Screenshot 2025-03-18 201319.png" style="width: 30%;"/>

The set of possible actions from all states is {up, down, right} (left is not available).  
In programming terms, \[(-1, 0), (1, 0), (0, 1)] is the set of possible actions from all states.

While these actions are available in all states, they do not always affect the robot's state.  For example, in state (0,1) the (1, 0) "right" action can be executed but the robot will remain in state (0,1).  The situation is similar at the borders, for example in state (0,2) executing (0,1) "down" leaves the robot in state (0,2).

```python
def future_state(self, y, x, dy, dx):
        """Return the future state given the current state and action."""
        new_y = y + dy
        new_x = x + dx
        if new_y < 0 or new_y >= self.size[0] or new_x < 0 or new_x >= self.size[1]:
            new_y = y
            new_x = x
        if self.occupancy[new_y, new_x] == 1:
            new_y = y
            new_x = x
        return new_y, new_x
```

Our model of this system is deterministic.  This means that the probabilities for all state transitions are either 0 or 1.

Executing any action in state (3, 0) will result in a reward of +10.  Executing any action in any other state gives a reward of 0.

<img src= "docs/Screenshot 2025-03-18 201343.png" style="width: 30%;"/>

