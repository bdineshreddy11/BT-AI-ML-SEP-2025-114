import os

# Corrected project structure
project_structure = {
    "Task1-Titanic": {
        "files": ["titanic_analysis.py", "titanic_analysis.ipynb",
                  "best_decision_tree_model.pkl", "titanic_eda.png",
                  "confusion_matrix.png", "roc_curve.png",
                  "requirements.txt", "README.md"],
        "data": {
            "files": ["train.csv", "test.csv"]
        }
    }
}

def create_structure(base_path, structure):
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        # Create files
        for file_name in contents.get("files", []):
            file_path = os.path.join(folder_path, file_name)
            open(file_path, 'a').close()

        # Recursively create subfolders
        for subfolder, subcontents in contents.items():
            if subfolder != "files":
                create_structure(folder_path, {subfolder: subcontents})

# Create the project structure in current directory
create_structure(".", project_structure)

print("Project structure created successfully!")
