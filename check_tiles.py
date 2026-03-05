import onnx

model = onnx.load('export/batdetect2_model.onnx')
tiles = [node.name for node in model.graph.node if node.op_type == 'Tile']
print(f'Found {len(tiles)} Tile operations')
if tiles:
    print('Operations:', tiles)
else:
    print('No Tile operations - model is compatible with ESPDL!')
