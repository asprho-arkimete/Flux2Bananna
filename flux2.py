import json

with open('key.json', 'r') as f:
    config = json.load(f)

key = config['api_key']
print(key)  # Stampa: sk-1234567890abcdef


import torch
from diffusers import Flux2KleinPipeline
from optimum.quanto import freeze, qfloat8, quantize
from huggingface_hub import login
from PIL import Image
import warnings
import os
from deep_translator import GoogleTranslator

warnings.filterwarnings('ignore', category=UserWarning, message='.*peft_config.*')

os.makedirs("model",exist_ok=True)
os.makedirs("lora",exist_ok=True)


login(token=key)

def flux_2():
    global steps, risoluzione, text, Lora1, Lora2, model, img, img2, img3, scale_lora1, scale_lora2,seed_g,lab_seed_attuale
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Ottieni il modello selezionato
    model_name = model.get()
    if model_name == 'FLUX.2-klein-9B':
        path_model = "black-forest-labs/FLUX.2-klein-9B"
    else:
        path_model = f"model/{model_name}"
    
    print(f"Caricamento pipeline FLUX.2 path: {path_model}")
    pipe = Flux2KleinPipeline.from_pretrained(
        path_model, 
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )

    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = None

    # Carica le LoRA selezionate
    lora1_name = Lora1.get()
    lora2_name = Lora2.get()
    
    adapter_names = []
    adapter_weights = []
    
    if lora1_name != 'no_lora':
        lora1_path = f"./lora/{lora1_name}"
        if os.path.exists(lora1_path):
            print(f"Caricamento LoRA 1: {lora1_name}")
            try:
                pipe.load_lora_weights(lora1_path, adapter_name="lora1")
                adapter_names.append("lora1")
                adapter_weights.append(float(scale_lora1.get()))
                print(f"✓ LoRA 1 caricata con peso: {float(scale_lora1.get()):.2f}")
            except Exception as e:
                print(f"✗ Errore nel caricamento LoRA 1: {e}")
    
    if lora2_name != 'no_lora':
        lora2_path = f"./lora/{lora2_name}"
        if os.path.exists(lora2_path):
            print(f"Caricamento LoRA 2: {lora2_name}")
            try:
                pipe.load_lora_weights(lora2_path, adapter_name="lora2")
                adapter_names.append("lora2")
                adapter_weights.append(float(scale_lora2.get()))
                print(f"✓ LoRA 2 caricata con peso: {float(scale_lora2.get()):.2f}")
            except Exception as e:
                print(f"✗ Errore nel caricamento LoRA 2: {e}")
    
    # Attiva le LoRA se presenti
    if adapter_names:
        print(f"Attivazione LoRA: {adapter_names}")
        if len(adapter_names) == 1:
            pipe.set_adapters(adapter_names[0], adapter_weights=adapter_weights[0])
        else:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    print("Quantizzazione transformer...")
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)

    print("Quantizzazione text encoder...")
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        quantize(pipe.text_encoder, weights=qfloat8)
        freeze(pipe.text_encoder)

    if hasattr(pipe, 'vae') and pipe.vae is not None:
        quantize(pipe.vae, weights=qfloat8)
        freeze(pipe.vae)

    print("Abilitazione ottimizzazioni memoria...")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    torch.cuda.empty_cache()

    # Ottieni il prompt dall'interfaccia
    prompt = text.get("1.0", "end-1c").strip()
    if not prompt or prompt == "Inserisci un prompt":
        print("⚠ Nessun prompt inserito, uso prompt di default")
        prompt = "a beautiful portrait, photorealistic, high detail"
    
    # Ottieni la risoluzione
    res = risoluzione.get().split('x')
    width = int(res[0])
    height = int(res[1])
    
    # Ottieni gli steps
    num_steps = int(steps.get())
    
    # Traduci il prompt
    try:
        prompt_en = GoogleTranslator(source='it', target='en').translate(prompt)
        print(f"Prompt ITA: {prompt}")
        print(f"Prompt ENG: {prompt_en}")
    except Exception as e:
        print(f"⚠ Errore traduzione, uso prompt originale: {e}")
        prompt_en = prompt
    
    print(f"Risoluzione: {width}x{height}")
    print(f"Steps: {num_steps}")
    
    # Raccogli le immagini caricate
    images = []
    if img is not None:
        images.append(img)
        print("✓ Immagine 1 caricata")
    if img2 is not None: 
        images.append(img2)
        print("✓ Immagine 2 caricata")
    if img3 is not None:
        images.append(img3)
        print("✓ Immagine 3 caricata")

    print("Generazione immagine...")
    
    # Parametri di generazione
    # Gestione seed
    seed_value = seed_g.get()
    if seed_value == '-1':
        # Genera seed casuale
        seed_actual = torch.randint(0, 2**32, (1,)).item()
        seed_g.set(str(seed_actual))  # Aggiorna l'entry con il seed generato
    else:
        seed_actual = int(seed_value)
    
    # Aggiorna la label con il seed attuale
    lab_seed_attuale.config(text=f"Seed Attuale: {seed_actual}")
    
    print(f"Usando seed: {seed_actual}")
    
    # Parametri di generazione
    gen_params = {
        "prompt": prompt_en,
        "height": height,
        "width": width,
        "guidance_scale": 1.0,
        "num_inference_steps": num_steps,
        "generator": torch.Generator(device=device).manual_seed(seed_actual)
    }
        
    # Aggiungi immagini se presenti
    if len(images) > 0:
        # Se c'è una sola immagine, passa direttamente
        if len(images) == 1:
            gen_params["image"] = images[0]
        else:
            # Se ci sono più immagini, passa la lista
            gen_params["image"] = images
        print(f"✓ Usando {len(images)} immagine/i di riferimento")
    
    image = pipe(**gen_params).images[0]
    os.makedirs("out", exist_ok=True)
    
    # Trova il prossimo nome file disponibile
    k = 0
    while True:
        if k == 0:
            output_path = "out/flux2_klein.png"
        else:
            output_path = f"out/flux2_klein_{k}.png"
        
        if not os.path.exists(output_path):
            break
        k += 1

    image.save(output_path)
    print(f"✓ Fatto! Immagine salvata come {output_path}")
    
    # Mostra l'immagine generata nell'interfaccia
    try:
        global photo
        output_img = image.resize((dim, dim), Image.BICUBIC)
        photo = ImageTk.PhotoImage(output_img)
        frame.delete("all")
        frame.create_image(dim//2, dim//2, image=photo)
        frame.update()
    except Exception as e:
        print(f"⚠ Errore visualizzazione risultato: {e}")

from PIL import Image, ImageTk
import tkinterdnd2 as tkdnd
from tkinter import Canvas, Frame
import tkinter as tk

img = None
img2 = None
img3 = None
photo = None
photo2 = None
photo3 = None
dim = 512
dim2 = 256

def load(event, frame, img_var, photo_var, canvas_dim):
    global img, img2, img3, photo, photo2, photo3
    try:
        # Rimuovi le graffe {} che tkinterdnd2 aggiunge ai percorsi
        filepath = event.data.strip('{}')
        
        loaded_img = Image.open(filepath)
        w, h = loaded_img.size
        rw, rh = canvas_dim, canvas_dim
        
        if w > canvas_dim or h > canvas_dim:
            if w >= h:
                rh = (rw * h) // w
            else:
                rw = (rh * w) // h
        else:
            # Mantieni dimensioni originali se l'immagine è più piccola
            rw, rh = w, h
        
        loaded_img = loaded_img.resize((rw, rh), Image.BICUBIC)
        
        # Salva l'immagine nella variabile globale corretta
        if img_var == 'img':
            img = loaded_img
            photo = ImageTk.PhotoImage(loaded_img)
            current_photo = photo
        elif img_var == 'img2':
            img2 = loaded_img
            photo2 = ImageTk.PhotoImage(loaded_img)
            current_photo = photo2
        elif img_var == 'img3':
            img3 = loaded_img
            photo3 = ImageTk.PhotoImage(loaded_img)
            current_photo = photo3
        
        frame.delete("all")
        frame.create_image(canvas_dim//2, canvas_dim//2, image=current_photo)
        frame.update()
    except Exception as e:
        print(f"Errore: {e}")

from tkinter import Button, Text, ttk, Label, Frame as tkFrame

window = tkdnd.Tk()
window.resizable(False, False)
window.title("Flux_2")
window.geometry("1028x1000")
window.config(bg='gray')

frame_canvas = Frame(window, bg='gray')
frame_canvas.grid(row=0, column=0, padx=10, pady=10)

# Frame 1 - Grande a sinistra
frame = Canvas(frame_canvas, width=dim, height=dim, bg='red')
frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
frame.create_text(dim//2, dim//2, text="Trascina qui immagine di riferimento 1", 
                  fill="white", font=("Arial", 16))

frame.drop_target_register(tkdnd.DND_FILES)
frame.dnd_bind('<<Drop>>', lambda event: load(event, frame, 'img', 'photo', dim))

# Frame 2 - Piccolo in alto a destra
frame2 = Canvas(frame_canvas, width=dim2, height=dim2, bg='pink')
frame2.grid(row=0, column=1, padx=5, pady=5, sticky='n')
frame2.create_text(dim2//2, dim2//2, text="Immagine 2", 
                  fill="white", font=("Arial", 12))

frame2.drop_target_register(tkdnd.DND_FILES)
frame2.dnd_bind('<<Drop>>', lambda event: load(event, frame2, 'img2', 'photo2', dim2))

# Frame 3 - Piccolo in basso a destra
frame3 = Canvas(frame_canvas, width=dim2, height=dim2, bg='lightblue')
frame3.grid(row=1, column=1, padx=5, pady=5, sticky='n')
frame3.create_text(dim2//2, dim2//2, text="Immagine 3", 
                  fill="white", font=("Arial", 12))

frame3.drop_target_register(tkdnd.DND_FILES)
frame3.dnd_bind('<<Drop>>', lambda event: load(event, frame3, 'img3', 'photo3', dim2))

run_flux2 = Button(window, text='run_flux', bg='lightgreen', width=15, height=2, command=flux_2)
run_flux2.grid(row=0, column=1, sticky='ne', padx=10, pady=10)
text = Text(window, width=50, height=10)
text.grid(row=1, column=0, sticky='w', padx=100, pady=10)
text.insert('1.0', "Inserisci un prompt")

controlli = tkFrame(window, bg='gray')
controlli.grid(row=2, column=0, sticky='w')

import os

def load_model(dir):
    global model, Lora1, Lora2
    modelli = []
    if os.path.exists(dir):
        if dir == 'lora':
            modelli = ['no_lora']
        else:
            modelli=['FLUX.2-klein-9B']
        modelli = modelli + [d for d in os.listdir(dir)]
        if dir == 'lora':
            Lora1['values'] = modelli
            Lora2['values'] = modelli
            Lora1.current(0)
            Lora2.current(0)
        else:
            model['values'] = modelli
            model.current(0)

lab_model = Label(controlli, text='Modello', bg='gray')
lab_model.grid(row=0, column=0, sticky='w', padx=5, pady=5)
model = ttk.Combobox(controlli)
model.grid(row=1, column=0, sticky='w', padx=5, pady=5)
model.bind('<Button-1>', lambda e: load_model("model"))

lab_lora1 = Label(controlli, text='Lora 1', bg='gray')
lab_lora1.grid(row=0, column=1, sticky='w', padx=5, pady=5)
Lora1 = ttk.Combobox(controlli)
Lora1.grid(row=1, column=1, sticky='w', padx=5, pady=5)
Lora1.bind('<Button-1>', lambda e: load_model("lora"))

lab_lora2 = Label(controlli, text='Lora 2', bg='gray')
lab_lora2.grid(row=0, column=2, sticky='w', padx=5, pady=5)
Lora2 = ttk.Combobox(controlli)
Lora2.grid(row=1, column=2, sticky='w', padx=5, pady=5)
Lora2.bind('<Button-1>', lambda e: load_model("lora"))

# Carica i modelli all'avvio
load_model("model")
load_model("lora")

# Funzione per aggiornare la label degli steps
def update_steps(value):
    lab_steps.config(text=f'steps: {int(float(value))}')

# Funzione per aggiornare la label del LoRA 1
def update_lora1(value):
    lab_scale_lora1.config(text=f'LoRA 1: {float(value):.2f}')

# Funzione per aggiornare la label del LoRA 2
def update_lora2(value):
    lab_scale_lora2.config(text=f'LoRA 2: {float(value):.2f}')

# Steps
lab_steps = Label(controlli, text='steps: 8', bg='gray')
lab_steps.grid(row=2, column=0, padx=5, pady=5)

steps = ttk.Scale(controlli, from_=1, to=50, length=200, command=update_steps)
steps.set(8)
steps.grid(row=3, column=0, padx=5, pady=5)

# LoRA 1 Scale
lab_scale_lora1 = Label(controlli, text='LoRA 1: 0.80', bg='gray')
lab_scale_lora1.grid(row=2, column=1, padx=5, pady=5)

scale_lora1 = ttk.Scale(controlli, from_=0.0, to=1.0, length=200, command=update_lora1)
scale_lora1.set(0.80)
scale_lora1.grid(row=3, column=1, padx=5, pady=5)

# LoRA 2 Scale
lab_scale_lora2 = Label(controlli, text='LoRA 2: 0.80', bg='gray')
lab_scale_lora2.grid(row=2, column=2, padx=5, pady=5)

scale_lora2 = ttk.Scale(controlli, from_=0.0, to=1.0, length=200, command=update_lora2)
scale_lora2.set(0.80)
scale_lora2.grid(row=3, column=2, padx=5, pady=5)

# Risoluzione
lab_risoluzione = Label(controlli, text='Risoluzione', bg='gray')
lab_risoluzione.grid(row=4, column=0, sticky='w', padx=5, pady=5)

# Risoluzioni comuni fino a 2K
risoluzioni = [
    '512x512',   # SD standard
    '768x768',   # SD upscaled
    '960x960',   # Square HD
    '960x1024',  # Portrait
    '1024x960',  # Landscape
    '1024x1024', # Square Full HD
    '1024x1280', # Portrait HD
    '1280x1024', # Landscape HD
    '1280x1280', # Square 1.2K
    '1280x1536', # Portrait 1.5K
    '1536x1280', # Landscape 1.5K
    '1536x1536', # Square 1.5K
    '1536x2048', # Portrait 2K
    '2048x1536', # Landscape 2K
    '2048x2048'  # Square 2K
]

risoluzione = ttk.Combobox(controlli, values=risoluzioni, width=15)
risoluzione.set('1024x1024')
risoluzione.grid(row=5, column=0, sticky='w', padx=5, pady=5)

# seed
lab_seed = Label(controlli, text='Seed Generate', bg='gray')
lab_seed.grid(row=4, column=1, sticky='w', padx=5, pady=5)

lab_seed_attuale = Label(controlli, text='Seed Attuale: None', bg='gray')
lab_seed_attuale.grid(row=4, column=2, sticky='w', padx=5, pady=5)

seed_g = tk.StringVar()
seed_g.set('-1')
seed_entry = tk.Entry(controlli, textvariable=seed_g)
seed_entry.grid(row=5, column=1, sticky='w', padx=5, pady=5)

from tkinter import messagebox

from tkinter import messagebox
import os

def f_face_swap():
    global var_face,img
    if var_face.get():
        if os.path.exists("out"):
            ultima_image = None
            k = 1
            
            # Cerca l'ultima immagine partendo da flux2_klein.png
            if os.path.exists("out/flux2_klein.png"):
                ultima_image = "out/flux2_klein.png"
            
            # Continua a cercare flux2_klein_1.png, flux2_klein_2.png, ecc.
            while True:
                test_path = f"out/flux2_klein_{k}.png"
                if os.path.exists(test_path):
                    ultima_image = test_path  # Aggiorna con l'ultima trovata
                    k += 1
                else:
                    break  # Quando non trova più, esce
            
            # Mostra dialog
            if ultima_image:
                risposta = messagebox.askyesno(
                    "Face Swapping", 
                    f"Vuoi applicare Face Swapping all'immagine:\n{ultima_image}?"
                )
                
                if risposta:
                    print(f"Applico Face swap a: {ultima_image}")
                    
                    if img is not None:
                        import shutil
                        import time
                        
                        # Prepara i file
                        output_filename = os.path.basename(ultima_image)
                        
                        # Copia target
                        shutil.copyfile(ultima_image, "roop/target.jpg")
                        print("✓ Copiato target")
                        
                        # Salva source
                        img.save("roop/input.jpg")
                        print("✓ Salvata source")
                        
                        # Esegui Roop (tutto in un comando)
                        comando = f"cd roop && python run.py --source input.jpg --target target.jpg --output {output_filename}"
                        os.system(comando)
                        
                        # Aspetta completamento
                        time.sleep(2)
                        
                        # Verifica se il file esiste
                        roop_output = f"roop/{output_filename}"
                        if os.path.exists(roop_output):
                            # Sposta il risultato
                            final_path = ultima_image.replace('.png', '_swp_face.png')
                            shutil.move(roop_output, final_path)
                            print(f"✅ Salvato: {final_path}")
                        else:
                            print(f"❌ File non trovato: {roop_output}")
                        
                        # Cleanup
                        try:
                            os.remove("roop/input.jpg")
                            os.remove("roop/target.jpg")
                        except:
                            pass
                    else:
                        print("❌ Nessuna immagine sorgente!")




                else:
                    print("Face swap annullato")
            else:
                messagebox.showwarning("Attenzione", "Nessuna immagine trovata!")

# Crea la variabile Boolean
var_face = tk.BooleanVar()
var_face.set(False)

# Crea la checkbox
face_swap = tk.Checkbutton(
    controlli, 
    text="Face Swapping", 
    variable=var_face,
    bg='gray',
    command=f_face_swap  # ← Usa command invece di bind
)
face_swap.grid(row=5, column=2, sticky='w', padx=5, pady=5)


window.mainloop()