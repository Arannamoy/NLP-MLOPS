import gradio as gr

def predict(class_idx):
    class_idx=int(class_idx)
    if class_idx==0:
        return "Cat"
    if class_idx==1:
        return "Dog"
    if class_idx==2:
        return "Person"
    else:
        return "No Class Label Found."
    
demo=gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
)


if __name__=="__main__":
    demo.launch(share=True)