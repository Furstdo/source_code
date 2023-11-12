from dataclasses import dataclass

@dataclass
class FooWrapper:
    Battery:            int

# save an object into a list
l = []
obj = FooWrapper(5)
l.append(obj)

# change the value of the initial object
obj.value = 3
print(l[0].value) # prints 3 since it's still the same reference
obj.value = 9
print(l[0].value) # prints 3 since it's still the same reference
