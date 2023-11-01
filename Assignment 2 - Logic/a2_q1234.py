import re

# Add the code for this function as described in the notebook 
# def evaluate(s):
def evaluate(s):
    operator = s.split(' ')[0]
    operand1 = int(s.split(' ')[1])
    operand2 = int(s.split(' ')[2])

    if operator == '&':
        result = operand1 & operand2
    elif operator == '=>':
        result = 1 if operand1 <= operand2 else 0
    elif operator == '<=>':
        result = 1 if operand1 == operand2 else 0
    elif operator == '|':
        result = operand1 | operand2
    return result

# Examples test cases
e1 = "| 0 1"
e2 = "<=> 1 1"
e3 = "& 0 0"

res_e1 = evaluate(e1)
res_e2 = evaluate(e2)
res_e3 = evaluate(e3)


print(f'{e1} = {res_e1}')
print(f'{e2} = {res_e2}')
print(f'{e3} = {res_e3}')

d = {'foo': 0, 'b': 1}
print(d)
be1 = '& 0 1'
be2 = '& foo 1'
be3 = '& foo ~b'


# Add the code for this function 
# def evaluate_with_bindings(s,d):
def evaluate_with_bindings(s, d):
    for k, v in d.items():
        if k in s:
            s = s.replace(k, str(v))

    while '~' in s:
        index = s.rfind('~')
        value = int(not int(s[index + 1]))
        s = s[:index] + str(value) + s[index + 2:]

    return evaluate(s)

# Example test cases 
res_be1 = evaluate_with_bindings(be1,d)
res_be2 = evaluate_with_bindings(be2,d)
res_be3 = evaluate_with_bindings(be3,d)

print(f'{be1} = {res_be1}')
print(f'{be2} = {res_be2}')
print(f'{be3} = {res_be3}')


# Add the code for this function as described in the notebook 
# You can add helper functions if you want as long as the function works as expected 

# def prefix_eval(input_str, d): 
def replace(input_str, d):
  for k, v in d.items():
        if k in input_str:
            input_str = input_str.replace(k, str(v))
  return input_str

def prefix_eval(input_str, d):
    input_str = replace(input_str, d)

    tokens = re.findall(r'(\(|\)|\w+|\||=>|&|~)', input_str)

    def recurse(tokens):
        if not tokens:
            return None

        token = tokens.pop(0)

        if '~' in token:
          value = recurse(tokens)
          return int(not int(value))

        if token in ['&', '|', '=>', '<=>']:
            value1 = recurse(tokens)
            value2 = recurse(tokens)

            if token == '&':
                return value1 and value2
            elif token == '|':
                return value1 or value2
            elif token == '=>':
                return 1 if value1 <= value2 else 0
            elif token == '<=>':
                return 1 if value1 == value2 else 0
        else:
            return int(token)

    result = recurse(tokens)
    return result

d = {"a": 1, "b": 0}
pe1 = "& a | 0 1"
pe2 = "& 0 | 1 b"
pe2 = "| 1 => ~b b"
pe3 = "<=> b <=> ~b 0"
pe4 = "=> 1 & a 0"
pe5 = "& ~a <=> 0 0"

print(d)
for e in [pe1,pe2,pe3,pe4,pe5]:
    print("%s \t = %d" % (e, prefix_eval(e,d)))

### SAMPLE OUTPUT 
# | 0 1 = 1
# <=> 1 1 = 1
# & 0 0 = 0
# {'foo': 0, 'b': 1}
# & 0 1 = 0
# & foo 1 = 0
# & foo ~b = 0
# {'a': 1, 'b': 0}
# & a | 0 1        = 1
# | 1 => ~b b      = 1
# <=> b <=> ~b 0   = 1
# => 1 & a 0       = 0
# & ~a <=> 0 0     = 0



