import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import os
import copy
import time
import threading
from PIL import Image, ImageTk
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# Transform weights from PyTorch Documentation
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.current_image_path = None

        # credit to github.com/rdbende for the azure tkinter theme
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.tk.call("source", "azure.tcl")
        self.tk.call("set_theme", "dark")

        self.title("EasyCV")
        self.geometry("1024x600")

        left_frame = ttk.Frame(self)
        right_frame = ttk.Frame(self)

        left_frame.pack(side="left", fill="both", expand=True)
        right_frame.pack(side="right", fill="both", expand=True)

        ttk.Label(left_frame, text="EasyCV", font=("TkDefaultFont", 16)).pack(pady=10)

        ttk.Label(left_frame, text="Model Name").pack(pady=5)
        self.model_name_entry = ttk.Entry(left_frame)
        self.model_name_entry.pack(pady=5, fill='x', padx=10)
        
        ttk.Label(left_frame, text="Path to validation dataset").pack(pady=5)
        self.val_dataset_entry = ttk.Entry(left_frame)
        self.val_dataset_entry.pack(pady=5, fill='x', padx=10)
        ttk.Button(left_frame, text="Browse", command=self.browse_val_dataset).pack(pady=5, fill='x', padx=10)

        ttk.Label(left_frame, text="Device selection").pack(pady=5)
        self.device_options = ['CPU'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.device_dropdown = ttk.Combobox(left_frame, values=self.device_options, state="readonly")
        self.device_dropdown.current(0)  # Set the default selection to CPU
        self.device_dropdown.pack(pady=5, fill='x', padx=10)
        
        ttk.Label(left_frame, text="Number of epochs").pack(pady=5)
        self.epochs_entry = ttk.Entry(left_frame)
        self.epochs_entry.pack(pady=5, fill='x', padx=10)

        start_training_button = ttk.Button(left_frame, text="Start Training", command=self.start_training)
        start_training_button.pack(pady=5, fill='x', padx=10)

        self.progress_bar = ttk.Progressbar(left_frame)
        self.progress_bar.pack(pady=5, fill='x', padx=10)

        ttk.Label(left_frame, text="Path to saved model").pack(pady=5)
        self.model_path_entry = ttk.Entry(left_frame)
        self.model_path_entry.pack(pady=5, fill='x', padx=10)
        ttk.Button(left_frame, text="Browse", command=self.browse_model_path).pack(pady=5, fill='x', padx=10)

        image_frame = ttk.Frame(right_frame, relief="solid")
        image_frame.pack(pady=5, fill='both', expand=True, padx=10)
        image_frame.pack_propagate(False)

        self.image_placeholder = ttk.Label(image_frame, text="Click here to select an image")
        self.image_placeholder.pack(padx=10, pady=10, fill='both', expand=True)
        self.image_placeholder.bind("<Button-1>", self.query_image)  # Making the label clickable

        self.prediction_label = ttk.Label(right_frame, text="Prediction: None", relief="solid")
        self.prediction_label.pack(pady=5, fill='x', padx=10)

        query_image_button = ttk.Button(right_frame, text="Query Image", command=self.use_current_image_for_query)
        query_image_button.pack(pady=5, fill='x', padx=10)
        self.minsize(1024, 600)

    def browse_val_dataset(self):
        directory = filedialog.askdirectory()
        if directory:
            self.val_dataset_entry.delete(0, tk.END)
            self.val_dataset_entry.insert(0, directory)

    def browse_model_path(self):
        filename = filedialog.askopenfilename(
            title="Select a model file",
            filetypes=(("PyTorch Model", "*.pth"), ("All Files", "*.*"))
        )
        if filename:
            self.model_path_entry.delete(0, tk.END)
            self.model_path_entry.insert(0, filename)


    def start_training(self):
        model_name = self.model_name_entry.get()
        val_path = self.val_dataset_entry.get()
        device_selection = self.device_dropdown.get()
        epochs = int(self.epochs_entry.get())
        if device_selection == 'CPU':
            device = torch.device('cpu')
        else:
            device = torch.device(device_selection)
        training_thread = threading.Thread(target=self.run_training, args=(model_name, val_path, device, epochs))
        training_thread.start()

    def run_training(self, model_name, val_path, device, epochs):
        def update_progress(progress):
            self.progress_bar['value'] = progress
            self.update_idletasks()
        # Here, you should modify the train function to handle model naming and saving appropriately.
        train(model_name, val_path, device, epochs, update_progress)
        messagebox.showinfo("Training Complete", f"The model has finished training and is saved as {model_name}.")

    def query_image(self, event):
        """Handles image selection and display without prediction."""
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*"))
        )
        if filepath:
            self.current_image_path = filepath  # Store the current image path
            img = Image.open(filepath)
            img.thumbnail((self.image_placeholder.winfo_width(), self.image_placeholder.winfo_height()))
            photo = ImageTk.PhotoImage(img)
            self.image_placeholder.configure(image=photo)
            self.image_placeholder.image = photo  # Keep a reference

    def use_current_image_for_query(self):
        if self.current_image_path:
            self.predict_image_class(self.current_image_path)
        else:
            messagebox.showinfo("Info", "Please select an image first.")

    def predict_image_class(self, image_path):
        # Load model and class names
        model_path = self.model_path_entry.get()
        if not model_path:
            messagebox.showerror("Error", "Please specify the path to the saved model.")
            return
        
        model_info = torch.load(model_path, map_location=torch.device('cpu'))
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(model_info['class_names']))
        model.load_state_dict(model_info['model_state_dict'])
        model.eval()  # Set model to evaluation mode
        
        # Process image and make predictions
        image_tensor = process_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)
            class_name = model_info['class_names'][preds.item()]
            self.prediction_label.config(text=f"Prediction: {class_name}")

    #TODO: Display device name instead of CUDA #
    def generate_device_mapping():
        devices = ['CPU'] + [f'GPU {i}' for i in range(torch.cuda.device_count())]
        mapping = {device: 'cpu' if i == 0 else f'cuda:{i-1}' for i, device in enumerate(devices)}
        return mapping

#perform the required setup for training
def train(name, data_dir, device, num_epochs, update_progress):
    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), DATA_TRANSFORMS['train'])}
    
    if os.path.exists(os.path.join(data_dir, 'val')):
        image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), DATA_TRANSFORMS['val'])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in image_datasets.keys()}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    class_names = image_datasets['train'].classes
    
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    best_weights = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, dataloaders, dataset_sizes, device, update_progress)
    
    model_info = {
        'class_names': class_names,
        'model_state_dict': best_weights
    }
    torch.save(model_info, f'{name}.pth')

#perform the actual model training
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, update_progress):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            progress = (epoch + 1) / num_epochs * 100
            update_progress(progress)
    model.load_state_dict(best_model_wts)
    return best_model_wts

#Converts the submitted image to a tensor
def process_image(image_path):
    image = Image.open(image_path)
    preprocessing = DATA_TRANSFORMS['val']
    image_tensor = preprocessing(image).unsqueeze(0)
    return image_tensor


if __name__ == "__main__":
    app = App()
    app.mainloop()
