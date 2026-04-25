# Reinforcement Learning Agents for Connect Four

Repozytorium zawiera kod źródłowy wykorzystany w projekcie inżynierskim dotyczącym szkolenia agentów uczenia ze wzmocnieniem w strategicznej grze planszowej Connect Four.

Celem projektu jest porównanie działania algorytmów **PPO** oraz **MaskablePPO** w środowisku Connect Four, ze szczególnym uwzględnieniem wpływu maskowania nielegalnych akcji na efektywność uczenia agenta.

## Autor

Andrzej Paczos

Projekt wykonany w ramach pracy inżynierskiej.

---

## Opis projektu

W projekcie analizowane jest zadanie uczenia agenta grającego w Connect Four przeciwko przeciwnikowi wybierającemu losowe legalne ruchy. Porównywane są dwa podejścia:

- **PPO** — klasyczna implementacja algorytmu Proximal Policy Optimization z biblioteki `stable-baselines3`,
- **MaskablePPO** — rozszerzona implementacja PPO z biblioteki `sb3-contrib`, wykorzystująca maskowanie nielegalnych akcji.

Eksperymenty są prowadzone dla kilku ziaren losowości, a modele są okresowo zapisywane jako punkty kontrolne. Następnie każdy punkt kontrolny jest ewaluowany na ustalonej liczbie gier, a wyniki są zapisywane do plików CSV.

Głównym kryterium porównania jest liczba kroków środowiska potrzebna do osiągnięcia określonego progu skuteczności, mierzonego jako odsetek zwycięstw przeciwko losowemu przeciwnikowi.

---

## Środowisko eksperymentalne

Gra Connect Four została zaimplementowana na podstawie środowiska `connect_four_v3` z biblioteki PettingZoo.

W projekcie wykorzystywany jest wrapper przekształcający środowisko typu AEC do postaci zgodnej z interfejsem Gymnasium. Agent wykonuje ruchy w dyskretnej przestrzeni akcji odpowiadającej siedmiu kolumnom planszy.

### Podstawowe właściwości środowiska

| Właściwość | Wartość |
|---|---|
| Gra | Connect Four |
| Bibliotka środowiska | PettingZoo |
| Wrapper | Gymnasium-compatible wrapper |
| Przestrzeń akcji | `Discrete(7)` |
| Akcje | indeksy kolumn od `0` do `6` |
| Obserwacja | tensor planszy |
| Przeciwnik | losowy wybór legalnej akcji |
| Warunek końca gry | zwycięstwo, porażka, remis lub nielegalna akcja |
| Obsługa nielegalnej akcji | natychmiastowe zakończenie epizodu z karą |

---

## Struktura repozytorium

Struktura projektu:

```text
.
├── configs/
│   ├── train_ppo.yaml
│   ├── train_maskableppo.yaml
│   └── eval.yaml
│
├── envs/
│   └── connect4_wrapper.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── plot.py
│
├── results/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
├── plots/
│   └── .gitkeep
│
├── requirements.txt
├── requirements_exact.txt
├── reproduce.sh
└── README.md
```

Opis katalogów:

| Katalog / plik | Opis |
|---|---|
| `configs/` | Pliki konfiguracyjne YAML dla treningu i ewaluacji |
| `envs/` | Implementacja wrappera środowiska Connect Four |
| `scripts/` | Skrypty do treningu, ewaluacji i generowania wykresów |
| `results/` | Wyniki ewaluacji w formacie CSV |
| `models/` | Zapisane modele i punkty kontrolne |
| `logs/` | Logi treningu, np. TensorBoard |
| `plots/` | Wygenerowane wykresy |
| `requirements.txt` | Lista podstawowych zależności |
| `requirements_exact.txt` | Dokładne wersje bibliotek użyte w eksperymencie |
| `reproduce.sh` | Skrypt do odtworzenia eksperymentu |

---

## Wymagania

Projekt wymaga środowiska Python oraz bibliotek związanych z uczeniem ze wzmocnieniem.

Zalecana wersja Pythona:

```text
Python 3.11
```

Główne biblioteki:

- `gymnasium`
- `stable-baselines3`
- `sb3-contrib`
- `pettingzoo[classic]`
- `numpy`
- `pandas`
- `matplotlib`
- `tensorboard`
- `pygame`

Do dokładnej reprodukcji wyników należy używać pliku:

```text
requirements_exact.txt
```

---

## Instalacja

### 1. Sklonowanie repozytorium

```bash
git clone <adres-repozytorium>
cd <nazwa-repozytorium>
```

### 2. Utworzenie środowiska wirtualnego

```bash
python -m venv .venv
```

Aktywacja środowiska:

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

### 3. Instalacja zależności

Do zwykłego uruchomienia projektu:

```bash
pip install -r requirements.txt
```

Do reprodukcji eksperymentów:

```bash
pip install -r requirements_exact.txt
```

---

## Reprodukcja eksperymentów

Najprostszym sposobem odtworzenia eksperymentów jest uruchomienie skryptu:

```bash
bash reproduce.sh
```

Skrypt wykonuje pełny pipeline eksperymentalny:

1. tworzy lub aktywuje środowisko wirtualne,
2. instaluje zależności z dokładnymi wersjami,
3. uruchamia trening modeli,
4. zapisuje punkty kontrolne,
5. przeprowadza ewaluację,
6. zapisuje wyniki do plików CSV,
7. generuje wykresy.

W przypadku systemu Windows skrypt można uruchomić np. przez Git Bash.

---

## Trening modeli

Trening modelu PPO:

```bash
python scripts/train.py --config configs/train_ppo.yaml
```

Trening modelu MaskablePPO:

```bash
python scripts/train.py --config configs/train_maskableppo.yaml
```

Parametry treningu są przechowywane w plikach YAML, co pozwala oddzielić konfigurację eksperymentu od kodu źródłowego.

Przykładowe parametry konfiguracyjne:

```yaml
algo: PPO
total_timesteps: 1000000
seed: 0
checkpoint_freq: 50000
```

---

## Ewaluacja modeli

Ewaluacja zapisanych punktów kontrolnych:

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Wyniki ewaluacji są zapisywane w katalogu:

```text
results/
```

Przykładowy format pliku wynikowego:

```csv
algo,seed,step,games,wins,win_rate,ci_low,ci_high
PPO,0,50000,100,62,0.620000,0.522087,0.708958
MaskablePPO,0,50000,100,75,0.750000,0.656956,0.824334
```

Znaczenie kolumn:

| Kolumna | Opis |
|---|---|
| `algo` | nazwa algorytmu |
| `seed` | ziarno losowości |
| `step` | liczba kroków środowiska |
| `games` | liczba rozegranych gier ewaluacyjnych |
| `wins` | liczba zwycięstw agenta |
| `win_rate` | odsetek zwycięstw |
| `ci_low` | dolne ograniczenie przedziału ufności Wilsona |
| `ci_high` | górne ograniczenie przedziału ufności Wilsona |

---

## Generowanie wykresów

Po zakończeniu ewaluacji można wygenerować wykresy:

```bash
python scripts/plot.py
```

Wykresy są zapisywane w katalogu:

```text
plots/
```

Typowy wykres przedstawia zależność skuteczności agenta od liczby kroków środowiska:

```text
win_rate vs environment steps
```

Dodatkowo na wykresie mogą być przedstawione przedziały ufności dla estymowanego odsetka zwycięstw.

---

## Metodyka porównania

Porównanie algorytmów opiera się na zasadach fair comparison:

- oba algorytmy są trenowane w tym samym środowisku,
- wykorzystywany jest ten sam przeciwnik ewaluacyjny,
- porównanie odbywa się dla tych samych budżetów kroków środowiska,
- ewaluacja jest prowadzona dla tych samych ziaren losowości,
- skuteczność jest mierzona jako odsetek zwycięstw przeciwko przeciwnikowi losowemu,
- niepewność estymacji jest opisywana za pomocą przedziału ufności Wilsona.

Głównym punktem końcowym eksperymentu jest liczba kroków środowiska potrzebna do osiągnięcia ustalonego progu skuteczności.

---

## PPO i MaskablePPO

### PPO

PPO, czyli Proximal Policy Optimization, jest algorytmem uczenia ze wzmocnieniem należącym do metod gradientu polityki. W projekcie wykorzystywana jest implementacja z biblioteki `stable-baselines3`.

### MaskablePPO

MaskablePPO jest wariantem PPO umożliwiającym maskowanie nielegalnych akcji. Dzięki temu agent nie wybiera akcji, które w danym stanie gry są niedozwolone.

W Connect Four maskowanie jest szczególnie istotne, ponieważ nie można wykonać ruchu w kolumnie, która jest już pełna.

---

## Wyniki

Wyniki eksperymentów są zapisywane w katalogu:

```text
results/
```

Wygenerowane wykresy znajdują się w katalogu:

```text
plots/
```

Przykładowe pliki wynikowe:

```text
results/eval_ppo.csv
results/eval_maskableppo.csv
plots/win_rate_vs_step.png
```

---

## Uwagi dotyczące reprodukowalności

W projekcie stosowane są następujące mechanizmy zwiększające reprodukowalność:

- jawnie ustawiane ziarna losowości,
- zapis konfiguracji eksperymentów w plikach YAML,
- zapis dokładnych wersji bibliotek w `requirements_exact.txt`,
- zapis punktów kontrolnych modeli,
- eksport wyników do plików CSV,
- centralny skrypt `reproduce.sh` odtwarzający eksperyment.

Należy jednak pamiętać, że wyniki eksperymentów uczenia ze wzmocnieniem mogą nieznacznie różnić się między uruchomieniami ze względu na niedeterminizm bibliotek numerycznych, systemu operacyjnego lub sprzętu.

---

## Przykładowy pełny workflow

```bash
# 1. Instalacja zależności
pip install -r requirements_exact.txt

# 2. Trening PPO
python scripts/train.py --config configs/train_ppo.yaml

# 3. Trening MaskablePPO
python scripts/train.py --config configs/train_maskableppo.yaml

# 4. Ewaluacja
python scripts/evaluate.py --config configs/eval.yaml

# 5. Wykresy
python scripts/plot.py
```

Alternatywnie:

```bash
bash reproduce.sh
```

---

## Licencja

Jeżeli w repozytorium znajduje się plik `LICENSE`, określa on zasady wykorzystania kodu.

W przypadku braku jawnie określonej licencji wszystkie prawa pozostają zastrzeżone przez autora.

---

## Cytowanie projektu

Jeżeli wykorzystujesz ten kod lub wyniki, proszę o odniesienie się do pracy inżynierskiej, w ramach której projekt został wykonany.

```text
Andrzej Paczos, Training reinforcement learning agents in strategic board games, engineering thesis, 2026.
```