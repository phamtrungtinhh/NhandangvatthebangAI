import ast, traceback
s=open('app.py','r',encoding='utf-8').read()
try:
    ast.parse(s)
    print('AST ok')
except Exception as e:
    print(repr(e))
    traceback.print_exc()
