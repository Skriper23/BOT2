# 🤖 Binance KNN Trading Bot

KNN bot. Indikatori (RSI, EMA, MACD)

---

## 📦 Instalacija

### 1. Instalacija packega za svaki distro 

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git

#### Fedora  
sudo dnf install python3 python3-pip python3-virtualenv git 

#### Arch 
sudo pacman -Syu python python-pip python-virtualenv git

ili sa AUR helperom tip yay:

yay -S python python-pip python-virtualenv git

Kod pacmana imati ukljucen multilib

### 2.Kloniranje repozitorija

git clone https://github.com/USERNAME/TradingBot.git
cd TradingBot

Postaviti virtual enviroment Arch sigurno nebude dal jer se packagi breakaju

Najjednostavnije: 
 
python3 -m venv venv
source venv/bin/activate

Različito za svaki distro ali uglavnom princip je isti

### 3.Python dependencies

Sa pip: 
pip install -r requirements.txt

Pip se dobije sa python installacijom ako nema isntall python3-pip ili python-virtualenv
# BOT2
