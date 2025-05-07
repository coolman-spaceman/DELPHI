#IMPORTING SUFF
######################################################################################
from flask import Flask, render_template, request, jsonify
import torch
from sklearn.manifold import TSNE
import torchvision.models as models
from flask import request, jsonify
from PIL import Image
from torchvision import transforms
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent RuntimeError in Flask
import matplotlib.pyplot as plt

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shutil
import torch.fx
######################################################################################

################################ DEFINE APP  #########################################
app = Flask(__name__)
loaded_model = None
activation_cache = {}

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Preprocessing for the model
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

#################################
# Load ImageNet class labels
with open("imagenet_class_index.json", "r") as f:
    idx_to_label = {int(k): v[1] for k, v in json.load(f).items()}
######################################################################################

@app.route('/')
def index():
    IMAGE_FOLDER = 'images'
    if os.path.exists(IMAGE_FOLDER):
        for filename in os.listdir(IMAGE_FOLDER):
            file_path = os.path.join(IMAGE_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return render_template('index.html')
######################################################################################
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Clear the folder
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

    # Save new image
    if 'image' not in request.files:
        return jsonify({'message': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return jsonify({'message': 'Image uploaded successfully'})

######################################################################################
@app.route('/load_model', methods=['POST'])
def load_model():
    global loaded_model
    model_name = request.json.get('model')
    
    if model_name == 'resnet50':
        loaded_model = models.resnet50(pretrained=True)
    elif model_name == 'resnet18':
        loaded_model = models.resnet18(pretrained=True)
    elif model_name == 'inception_v3':
        loaded_model = models.inception_v3(pretrained=True)
    else:
        return jsonify({'message': 'Unknown model'}), 400

    loaded_model.eval()
    print(model_name)
    return jsonify({'message': 'Model loaded'})
######################################################################################

@app.route('/layers')
def layers():
    if not loaded_model:
        return "Model not loaded", 400

    layer_info = []
    for name, module in loaded_model.named_modules():
        if name != "":
            detail_str = str(module)

            if len(detail_str) > 200:
                continue
            layer_info.append({
                'name': name,
                'type': module.__class__.__name__,
                'details': str(module)
            })

    #print(layer_info[0])
    return render_template('layers.html', layers=layer_info)

######################################################################################
@app.route('/predict', methods=['POST'])
def predict():
    if not loaded_model:
        return jsonify({'error': 'Model not loaded'}), 400

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image uploaded'}), 400

    # Save image
    image_path = os.path.join('images', image_file.filename)
    os.makedirs('images', exist_ok=True)
    image_file.save(image_path)

    # Preprocess image
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)  # shape (1, 3, 224, 224)

    # Prediction
    with torch.no_grad():
        output = loaded_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probabilities, 5)

    top5_results = [{
        'label': idx_to_label[int(idx)],
        'probability': float(prob)
    } for prob, idx in zip(top5.values, top5.indices)]

    return jsonify({'top_5': top5_results})
######################################################################################


@app.route('/activations/<layer_name>')
def show_activations(layer_name):
    if not loaded_model:
        return "Model not loaded", 400

    image_dir = 'images'
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        return "No image found. Please upload one.", 400

    image_path = os.path.join(image_dir, os.listdir(image_dir)[-1])
    input_image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    activation_cache.clear()

    def hook_fn(module, input, output):
        activation_cache['data'] = output.detach()

    target_layer = dict(loaded_model.named_modules()).get(layer_name)
    if not target_layer:
        return f"Layer '{layer_name}' not found.", 404

    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        try:
            loaded_model(input_tensor)
        except Exception as e:
            hook.remove()
            return f"Forward pass failed: {str(e)}", 500
    hook.remove()

    activations = activation_cache.get('data')
    if activations is None:
        return "No activations captured", 500

    act_np = activations.squeeze(0).cpu().numpy()
    output_dir = os.path.join('static', 'activations', layer_name)
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous outputs
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    image_paths = []
    ## min activations 20
    for i in range(min(20, act_np.shape[0])):
        act = act_np[i]
        act_min, act_max = act.min(), act.max()
        if act_max - act_min > 0:
            act_norm = (act - act_min) / (act_max - act_min)
        else:
            act_norm = np.zeros_like(act)

        fig, ax = plt.subplots()
        ax.imshow(act_norm, cmap='viridis')
        ax.axis('off')
        image_file = f'activation_{i+1}.png'
        image_path = os.path.join(output_dir, image_file)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        image_paths.append(f'/static/activations/{layer_name}/{image_file}')

    return render_template('activation.html', images=image_paths, layer_name=layer_name)

######################################################################################
@app.route('/delete_cache', methods=['POST'])
def delete_cache():
    cache_dir = os.path.join('static', 'activations')
    if os.path.exists(cache_dir):
        for subdir in os.listdir(cache_dir):
            path = os.path.join(cache_dir, subdir)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)
    return jsonify({'message': 'Activation cache deleted.'})
######################################################################################
################################   LIME    ###########################################

@app.route('/lime')
def lime_explanation():
    if not loaded_model:
        return "Model not loaded", 400

    image_dir = 'images'
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        return "No image found. Please upload one.", 400

    # Load uploaded image
    image_path = os.path.join(image_dir, os.listdir(image_dir)[-1])
    pil_image = Image.open(image_path).convert('RGB')
    np_image = np.array(pil_image)

    lime_output_dir = os.path.join('static', 'lime')
    if os.path.exists(lime_output_dir):
        shutil.rmtree(lime_output_dir)
    os.makedirs(lime_output_dir, exist_ok=True)

    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])

    def batch_predict(images):
        model_input = torch.stack([preprocess(Image.fromarray(img)).to(torch.float32) for img in images])
        with torch.no_grad():
            output = loaded_model(model_input)
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        return probs

    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

    explanation = explainer.explain_instance(
        np_image,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=500,####################################################################################
        segmentation_fn=segmentation_fn
    )

    top_label = explanation.top_labels[0]

    # Get the main explanation image (positive only)
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=100,
        hide_rest=False
    )

    # Also get the full heatmap image (positive & negative regions)
    heat_temp, heat_mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=100,
        hide_rest=False
    )

    from skimage.segmentation import mark_boundaries

    # Save explanation overlay
    fig1, ax1 = plt.subplots()
    ax1.imshow(mark_boundaries(temp, mask))
    ax1.axis('off')
    overlay_path = os.path.join(lime_output_dir, 'lime_overlay.png')
    fig1.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    # Save heatmap visualization
    fig2, ax2 = plt.subplots()
    ax2.imshow(mark_boundaries(heat_temp, heat_mask))
    ax2.axis('off')
    heatmap_path = os.path.join(lime_output_dir, 'lime_heatmap.png')
    fig2.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    return render_template(
        'lime.html',
        overlay_img='/static/lime/lime_overlay.png',
        heatmap_img='/static/lime/lime_heatmap.png'
    )

######################################################################################
def get_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No conv layer found")
#######################################################################################
def generate_gradcam(model, img_tensor, target_layer):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = torch.argmax(output)
    score = output[:, class_idx]
    model.zero_grad()
    score.backward()

    gradient = gradients[0]
    activation = activations[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    gradcam = torch.relu((weights * activation).sum(dim=1)).squeeze().cpu().detach().numpy()

    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    gradcam = np.uint8(255 * gradcam)

    handle_fw.remove()
    handle_bw.remove()

    return gradcam, class_idx.item()
####################################################################################
def save_gradcam_image(gradcam, orig_image, output_path):
    gradcam_resized = Image.fromarray(gradcam).resize(orig_image.size, Image.LANCZOS)
    gradcam_colored = np.array(gradcam_resized)
    cmap = plt.get_cmap('jet')
    heatmap = cmap(gradcam_colored / 255.0)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    blended = Image.blend(orig_image, Image.fromarray(heatmap), alpha=0.5)
    blended.save(output_path)
#####################################################################################

@app.route('/model_gradcam')
def model_gradcam():
    if not loaded_model:
        return "Model not loaded", 400
    model = loaded_model
    model.eval()

    image_dir = 'images'
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        return "No image found. Please upload one.", 400

    # Load uploaded image
    image_path = os.path.join(image_dir, os.listdir(image_dir)[-1])
    input_image = Image.open(image_path).convert('RGB')
    
    img_tensor = preprocess(input_image).unsqueeze(0)
    target_layer = get_last_conv_layer(model)
    gradcam, pred_class = generate_gradcam(model, img_tensor, target_layer)

    output_path = 'static/gradcam/gradcam_output.jpg'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_gradcam_image(gradcam, input_image, output_path)

    return render_template('gradcam_display.html', image_path=output_path)
##################################################################################
# DAG Netron model

def extract_graph_data(traced_graph):
    # Create example input and trace the model 
    nodes = []
    edges = []
    name_to_id = {}

    for idx, node in enumerate(traced_graph.graph.nodes):
        node_id = f"node_{idx}"
        name_to_id[node.name] = node_id
        #label = f"{node.target}" if node.op != 'placeholder' else f"Input: {node.target}"
        label = f"{node.name}"
        if node.op == 'output':
            label = "Output"
        nodes.append({"id": node_id, "label": label})

        # Add edges from args
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                edges.append({"from": name_to_id.get(arg.name, ""), "to": node_id})

        #print('Nodes: ', nodes[:12])
        #print('Edges: ', edges[:15])
    return {"nodes": nodes, "edges": edges}

@app.route("/get_graph")
def get_graph():
    if not loaded_model:
        return "Model not loaded", 400
    model = loaded_model
    model.eval()
    # Create example input and trace the model
    dummy_input = torch.randn(1, 3, 224, 224)
    traced = torch.fx.symbolic_trace(model)

    graph_data = extract_graph_data(traced)
    return jsonify(graph_data)

@app.route("/layers_visual")
def layers_visual():
    return render_template("layers_visual.html")  # frontend will fetch from /graph


#######################################################################################
#######################################################################################
if __name__ == '__main__':
    app.run(debug=True)
######################################################################################
