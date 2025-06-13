import os
import requests
import json
import time
from colorama import init, Fore, Style

# Initialiser colorama pour les couleurs dans le terminal
init()

# Configuration
MODEL = "mythomax:latest"  # Vous pouvez changer pour "mistral:latest" si vous préférez
OLLAMA_URL = "http://localhost:11434/api/generate"
HISTORY = []
SAVE_FILE = "save_game.json"

def clear_screen():
    """Efface l'écran du terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color=Fore.YELLOW):
    """Affiche du texte coloré"""
    print(f"{color}{text}{Style.RESET_ALL}")

def typing_effect(text, delay=0.01):
    """Affiche le texte avec un effet de machine à écrire"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def save_game():
    """Sauvegarde la partie en cours"""
    with open(SAVE_FILE, 'w', encoding='utf-8') as f:
        json.dump(HISTORY, f, ensure_ascii=False, indent=2)
    print_colored("Partie sauvegardée ✓", Fore.GREEN)

def load_game():
    """Charge une partie sauvegardée"""
    global HISTORY
    try:
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, 'r', encoding='utf-8') as f:
                HISTORY = json.load(f)
            print_colored("Partie chargée ✓", Fore.GREEN)
            return True
        return False
    except Exception as e:
        print_colored(f"Erreur lors du chargement: {str(e)}", Fore.RED)
        return False

def generate_response(prompt):
    """Génère une réponse via Ollama"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return json.loads(response.text)["response"]
        else:
            return f"Erreur: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Erreur de connexion à Ollama: {str(e)}"

def build_context():
    """Construit le contexte pour l'IA"""
    system_context = """Tu es un narrateur pour une aventure interactive dans l'univers de Mushoku Tensei.
L'utilisateur joue le rôle d'un personnage qui vient de naître avec les souvenirs de sa vie précédente.
Réponds de manière immersive en décrivant l'environnement, les sensations et les réactions.
À la fin de chaque réponse, propose toujours 3-4 actions possibles pour guider l'utilisateur.
Format pour les suggestions: "Tu peux: 1) Action 1, 2) Action 2, 3) Action 3"
Sois créatif et reste cohérent avec l'univers de Mushoku Tensei qui mélange fantasy, magie et réincarnation."""

    # Limiter l'historique à 10 échanges pour éviter les problèmes de contexte trop long
    relevant_history = HISTORY[-10:] if len(HISTORY) > 10 else HISTORY
    history_text = "\n".join(relevant_history)
    
    return f"{system_context}\n\n### Historique de la conversation:\n{history_text}"

def intro_sequence():
    """Séquence d'introduction du jeu"""
    clear_screen()
    
    print_colored("=" * 60, Fore.BLUE)
    print_colored("          MUSHOKU TENSEI - AVENTURE INTERACTIVE", Fore.CYAN)
    print_colored("=" * 60, Fore.BLUE)
    print_colored("\nBienvenue dans le monde de Mushoku Tensei!", Fore.WHITE)
    print_colored("Dans cette aventure, vous incarnez un personnage réincarné", Fore.WHITE)
    print_colored("avec les souvenirs de sa vie précédente.", Fore.WHITE)
    print_colored("\nVos choix détermineront votre destin dans ce monde fantastique.", Fore.WHITE)
    print_colored("\nCommandes spéciales:", Fore.CYAN)
    print_colored("  /aide - Affiche l'aide", Fore.CYAN)
    print_colored("  /sauvegarder - Sauvegarde la partie", Fore.CYAN)
    print_colored("  /charger - Charge une partie sauvegardée", Fore.CYAN)
    print_colored("  /quitter - Quitte le jeu", Fore.CYAN)
    print_colored("\nAppuyez sur Entrée pour commencer...", Fore.GREEN)
    input()
    
    clear_screen()
    intro_text = "La vie de Ethan commence tout juste dans le monde de Mushoku Tensei...\nTu ouvres les yeux pour la première fois. Le monde qui t'entoure est flou, rempli de formes et de couleurs indistinctes. Des voix douces murmurent autour de toi, et tu sens une chaleur réconfortante contre ta peau."
    typing_effect(intro_text)
    
    HISTORY.append("Narrateur: " + intro_text)
    
    # Premier prompt pour démarrer l'aventure
    first_prompt = """Tu viens de naître comme un bébé avec les souvenirs de ta vie précédente.
Le monde est flou autour de toi, mais tu reconnais que tu as été réincarné dans un corps de bébé.
Des voix parlent autour de toi, probablement tes parents.
Que ressens-tu et que souhaites-tu faire en premier dans ta nouvelle vie?"""
    
    return first_prompt

def process_command(command):
    """Traite les commandes spéciales commençant par /"""
    cmd = command.lower().strip()
    
    if cmd == "/aide" or cmd == "/help":
        print_colored("\n=== AIDE ===", Fore.CYAN)
        print_colored("  /aide - Affiche cette aide", Fore.CYAN)
        print_colored("  /sauvegarder - Sauvegarde la partie", Fore.CYAN)
        print_colored("  /charger - Charge une partie sauvegardée", Fore.CYAN)
        print_colored("  /quitter - Quitte le jeu", Fore.CYAN)
        return True
        
    elif cmd == "/sauvegarder" or cmd == "/save":
        save_game()
        return True
        
    elif cmd == "/charger" or cmd == "/load":
        load_game()
        return True
        
    elif cmd == "/quitter" or cmd == "/exit":
        print_colored("\nMerci d'avoir joué! À bientôt dans le monde de Mushoku Tensei!", Fore.CYAN)
        return "exit"
        
    return False

def main():
    """Fonction principale du jeu"""
    # Vérifier si une sauvegarde existe
    if not load_game():
        # Si pas de sauvegarde, démarrer une nouvelle partie
        first_prompt = intro_sequence()
    else:
        # Si une sauvegarde existe, montrer le dernier message
        if HISTORY:
            last_message = HISTORY[-1]
            print_colored("\nDernière action:", Fore.CYAN)
            print_colored(last_message, Fore.YELLOW)
        first_prompt = None
    
    # Boucle principale du jeu
    while True:
        if first_prompt:
            # Si c'est le début du jeu, utiliser le premier prompt
            context = build_context()
            response = generate_response(f"{context}\n\n{first_prompt}")
            print_colored("\n" + response, Fore.YELLOW)
            HISTORY.append("Narrateur: " + response)
            first_prompt = None
        else:
            # Demander action à l'utilisateur
            print_colored("\nQue souhaitez-vous faire?", Fore.GREEN)
            user_input = input("> ")
            
            # Traiter les commandes spéciales
            cmd_result = process_command(user_input)
            if cmd_result == "exit":
                break
            elif cmd_result:
                continue
            
            # Ajouter l'input utilisateur à l'historique
            HISTORY.append("Utilisateur: " + user_input)
            
            # Construire le contexte et générer la réponse
            context = build_context()
            full_prompt = f"{context}\n\nL'utilisateur fait: {user_input}\n\nNarration et suggestions:"
            
            print_colored("\nLe narrateur réfléchit...", Fore.CYAN)
            response = generate_response(full_prompt)
            
            # Afficher la réponse avec effet de machine à écrire
            print()
            typing_effect(response)
            
            # Ajouter la réponse à l'historique
            HISTORY.append("Narrateur: " + response)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nAu revoir!", Fore.CYAN)
    except Exception as e:
        print_colored(f"\nUne erreur est survenue: {str(e)}", Fore.RED)
