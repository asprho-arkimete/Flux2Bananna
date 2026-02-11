🍌 Flux2Bananna
Flux2Bananna è uno strumento avanzato per l'image editing basato sull'ecosistema Flux 2, ottimizzato per girare su hardware con bassa VRAM (Low VRAM) grazie alla quantizzazione del modello Klein-9B.

Il sistema è progettato per essere estremamente rapido, producendo risultati di alta qualità in soli 8 step, ed è completamente gratuito e open-source.

🚀 Caratteristiche Principali
Modello Core: Basato su Flux 2 Klein-9B (versione quantizzata).

Performance: Generazione ultra-veloce con soli 8 step di campionamento.

Low VRAM: Ottimizzato per schede grafiche consumer.

Customization: Supporto per Checkpoint personalizzati e fino a 2 LoRA aggiuntivi in contemporanea.

Editing Style: Workflow ispirato a "Nano Banana" per una manipolazione intuitiva delle immagini.

🛠️ Installazione
Clona il repository:

Bash
git clone https://github.com/asprho-arkimete/Flux2Bananna.git
cd Flux2Bananna
Crea l'ambiente virtuale (Python 3.10 consigliato):

Bash
python -m venv vflux2
# Attiva l'ambiente (Windows)
vflux2\Scripts\activate
# Attiva l'ambiente (Linux/Mac)
source vflux2/bin/activate
Installa le dipendenze:

Bash
pip install -r requirements.txt
Configurazione API Key: Crea un file chiamato config.json (o rinomina quello esistente) nella root del progetto e inserisci il tuo token di Hugging Face:

JSON
{
  "api_key": "LA_TUA_API_KEY_HUGGING_FACE"
}
💾 Download dei Modelli
Per far funzionare correttamente il software, segui questi passaggi su Civitai:

Checkpoint: Cerca modelli compatibili come Flux 2 Klein 9B o Flux 2 Klein 9B-Base.

Posiziona i file scaricati nella cartella /models.

LoRA: Scarica i tuoi LoRA preferiti e inseriscili nella cartella /loras.

📖 Come Utilizzarlo
Una volta configurato l'ambiente e inseriti i modelli, avvia l'interfaccia principale:

Bash
python flux2.py
(Assicurati di aver inserito i modelli corretti nelle rispettive cartelle prima dell'avvio).






