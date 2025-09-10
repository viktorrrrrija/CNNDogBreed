import torch
import torch.nn.functional as F
from torchvision import transforms, models
from safetensors.torch import load_file
from transformers import  AutoModelForImageClassification

#metoda za aplikaciju, predvidja klasu u yavisnosti od odabranog modela

def predict_image(img, class_names, model_type="custom"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    
    if model_type == "vit":
        model_name = "wesleyacheng/dog-breeds-multiclass-image-classification-with-vit"        
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model = model.to(device)
        
        use_class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    else:  # custom
        model = models.resnet18(pretrained=False)  # ne učitava pretrained težine
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # broj izlaza = broj klasa
        model = model.to(device)
        
        weights = load_file("model_weights.safetensors")
        model.load_state_dict(weights)
        
        
        use_class_names = class_names
        
        

    model.eval()
    threshold = 0.2
    with torch.no_grad():
        outputs = model(img_tensor)

        
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_idx = predicted.item()

   
    if confidence.item() >= threshold and predicted_idx < len(use_class_names):
        predicted_class = use_class_names[predicted_idx]
    else:
        predicted_class = "unknown"

    return predicted_class, confidence.item()
