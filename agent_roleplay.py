# ==========================================
# === MUSHOKU TENSEI ROLEPLAY AGENT ===
# === VERSION 3.0 - AVENTURE IMMERSIVE ===
# ==========================================
import os
import json
import time
import random
import datetime
import logging
import re
import math
import uuid
import hashlib
import copy
import shutil
import http.client
import urllib.parse
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque

# ===========================================
# === BLOC 1/12 : CONFIGURATION GÉNÉRALE ===
# ===========================================

# Gestion simplifiée des variables d'environnement
def load_env_variables():
    """Charge les variables d'environnement depuis un fichier .env si disponible"""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    
# Charger les variables d'environnement
try:
    load_env_variables()
    print("Variables d'environnement chargées.")
except Exception as e:
    print(f"Avertissement: Impossible de charger les variables d'environnement. {str(e)}")

# Client Ollama avec seulement la bibliothèque standard
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        parsed_url = urllib.parse.urlparse(base_url)
        self.host = parsed_url.netloc or "localhost:11434"
        
    def chat_completion(self, model, messages, max_tokens=None):
        """Alternative utilisant http.client (bibliothèque standard)"""
        conn = http.client.HTTPConnection(self.host)
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        headers = {'Content-type': 'application/json'}
        
        try:
            conn.request("POST", "/api/chat", json.dumps(payload), headers)
            response = conn.getresponse()
            data = response.read().decode()
            
            if response.status != 200:
                raise Exception(f"Erreur API: {response.status} - {data}")
            
            # Correction du traitement JSON
            try:
                # Essayer d'abord de parser tout le JSON
                return json.loads(data)
            except json.JSONDecodeError:
                try:
                    # Si échec, essayer de prendre seulement la première ligne JSON valide
                    first_json = data.strip().split('\n')[0]
                    return json.loads(first_json)
                except (json.JSONDecodeError, IndexError):
                    # Si toujours un échec, renvoyer une structure simplifiée
                    return {"response": data}
        finally:
            conn.close()

# Création du client
client = OllamaClient(base_url="http://localhost:11434")

# Configuration du système de logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"mushoku_rp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MushokuRP")

# Ajoute cette vérification après la configuration
try:
    response = client.chat_completion(
        model="mistral:latest",
        messages=[{"role": "system", "content": "Test de connexion à l'IA."}],
        max_tokens=5
    )
    logger.info("Connexion à l'IA établie avec succès")
    print("Connexion à l'IA établie avec succès!")
except Exception as e:
    logger.error(f"Erreur de connexion à l'IA: {str(e)}")
    print(f"Erreur: Impossible de se connecter à l'IA. Vérifiez qu'Ollama est en cours d'exécution avec mistral:latest.")
    print(f"Détails de l'erreur: {str(e)}")

# Variables globales de sauvegarde
SAVE_FILE = None
HISTORY_FILE = None
DEFAULT_MODEL = "mythomax:latest"  # Modèle par défaut (peut être changé dans les options)

# Version du jeu
VERSION = "3.0"

# Calendrier du monde de Mushoku Tensei
MOIS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
       "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
SAISONS = ["Hiver", "Hiver", "Printemps", "Printemps", "Printemps", "Été",
          "Été", "Été", "Automne", "Automne", "Automne", "Hiver"]

# Constantes pour les systèmes avancés
RACES = ["Humain", "Elfe", "Nain", "Beastfolk", "Migurd", "Supard", "Démon"]
ELEMENTS = ["Eau", "Feu", "Terre", "Vent", "Foudre", "Glace", "Lumière", "Ombre", "Temps", "Espace"]
RANGS_MAGIE = ["Débutant", "Intermédiaire", "Avancé", "Saint", "Roi", "Empereur", "Dieu"]
ECOLES_EPEE = ["Style Eau", "Style Nord", "Style Dieu-Épée"]

# Régions du monde
REGIONS = {
    "Continent Central": ["Royaume de Milis", "Saint Empire d'Asura", "Royaume de Shirone"],
    "Continent Démoniaque": ["Pays des Démons", "Terres Désolées"],
    "Continent des Bêtes": ["Forêt des Bêtes", "Grande Forêt"],
    "Continent des Cieux": ["Royaume Céleste"],
}

# Encodeur JSON personnalisé pour gérer les classes complexes
class EncodeurJeuPerso(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'asdict'):
            return asdict(obj)
        elif isinstance(obj, Enum):
            return obj.name
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

# Création des dossiers nécessaires
def creer_dossier_si_absent(dossier):
    if not os.path.exists(dossier):
        os.makedirs(dossier)
        logger.info(f"Dossier créé: {dossier}")

# Création des dossiers requis au démarrage
creer_dossier_si_absent("data")
creer_dossier_si_absent("backups")
creer_dossier_si_absent("historique")
creer_dossier_si_absent("personnages")
creer_dossier_si_absent("lore")

# ======================================================
# === BLOC 2/12 : STRUCTURES DE DONNÉES PERSONNAGES ===
# ======================================================

# Classes pour la météo et les conditions environnementales
class ConditionMeteo(Enum):
    ENSOLEILLE = auto()
    NUAGEUX = auto()
    PLUIE = auto()
    ORAGE = auto()
    NEIGE = auto()
    TEMPETE = auto()
    BRUME = auto()
    CANICULE = auto()
    CIEL_COUVERT = auto()
    CLAIR = auto()
    BROUILLARD = auto()

@dataclass
class ConditionsMeteo:
    condition: ConditionMeteo = ConditionMeteo.CLAIR
    temperature: float = 20.0
    intensite: int = 5  # Sur une échelle de 1-10
    duree_prevue: int = 8  # En heures

# Enums pour les caractéristiques
class Sexe(Enum):
    HOMME = auto()
    FEMME = auto()
    AUTRE = auto()

class Statut(Enum):
    VIVANT = auto()
    MORT = auto()
    INCONNU = auto()
    SCELLE = auto()

class Alignement(Enum):
    BON = auto()
    NEUTRE = auto()
    MAUVAIS = auto()
    CHAOTIQUE = auto()
    LOYAL = auto()

class RangAventurier(Enum):
    NONE = 0
    F = 1
    E = 2
    D = 3
    C = 4
    B = 5
    A = 6
    S = 7

# Structure pour les traits de personnalité
@dataclass
class TraitsPersonnalite:
    primaires: Dict[str, int] = field(default_factory=lambda: {})
    secondaires: Dict[str, int] = field(default_factory=lambda: {})
    cachés: Dict[str, int] = field(default_factory=lambda: {})
    traumatismes: List[str] = field(default_factory=list)
    valeurs: Dict[str, int] = field(default_factory=lambda: {})
    
    # Mémoire émotionnelle - stocke les réactions émotionnelles aux événements importants
    memoire_emotionnelle: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    
    def ajouter_souvenir_emotionnel(self, evenement: str, emotion: str, intensite: int, 
                                    contexte: str = "", date_jeu: Optional[str] = None):
        """Ajoute un souvenir émotionnel à la mémoire du personnage"""
        self.memoire_emotionnelle[evenement] = {
            "emotion": emotion,
            "intensite": intensite,
            "contexte": contexte,
            "date": date_jeu or "inconnu",
            "recurrence": 0,  # Nombre de fois que le personnage y repense
            "estompe": 0       # Degré d'estompage du souvenir (augmente avec le temps)
        }

# Structure pour les compétences et capacités
@dataclass
class Competence:
    nom: str
    niveau: int = 0
    experience: int = 0
    description: str = ""
    affinite: float = 1.0  # Multiplicateur d'apprentissage naturel
    usages: int = 0  # Nombre de fois utilisée
    derniere_evolution: Optional[datetime.datetime] = None
    
    # Pour le système d'évolution des compétences
    chemins_evolution: Dict[str, int] = field(default_factory=lambda: {})
    specialisation: Optional[str] = None

@dataclass
class CompetenceMagique(Competence):
    element: str = ""
    rang: str = "Débutant"
    incantation_requise: bool = True
    cout_mana: int = 0
    
    # Pour l'évolution magique basée sur l'usage
    style_usage: Dict[str, int] = field(default_factory=lambda: {
        "offensif": 0, 
        "défensif": 0, 
        "utilitaire": 0, 
        "soutien": 0
    })

# Structure pour les relations entre personnages
@dataclass
class Relation:
    cible_id: str
    nom_cible: str
    affinite: int = 0  # De -100 à 100
    confiance: int = 0
    respect: int = 0
    peur: int = 0
    dette: int = 0
    familiarite: int = 0
    historique: List[Dict[str, Any]] = field(default_factory=list)
    type_relation: Optional[str] = None  # ami, famille, mentor, etc.
    
    def modifier_relation(self, aspect: str, valeur: int, raison: str = ""):
        """Modifie un aspect spécifique de la relation et enregistre l'historique"""
        if hasattr(self, aspect):
            setattr(self, aspect, max(-100, min(100, getattr(self, aspect) + valeur)))
            
            self.historique.append({
                "date": datetime.datetime.now().isoformat(),
                "aspect": aspect,
                "modification": valeur,
                "raison": raison
            })

# Structure principale des personnages
@dataclass
class Personnage:
    id: str
    nom: str
    titre: Optional[str] = None
    race: str = "Humain"
    sexe: Sexe = Sexe.HOMME
    age: int = 20
    date_naissance: Optional[str] = None
    statut: Statut = Statut.VIVANT
    alignement: Alignement = Alignement.NEUTRE
    description: str = ""
    histoire: str = ""
    
    # Caractéristiques évolutives
    niveau: int = 1
    experience: int = 0
    rang_aventurier: RangAventurier = RangAventurier.NONE
    sante_max: int = 100
    sante_actuelle: int = 100
    mana_max: int = 50
    mana_actuel: int = 50
    endurance_max: int = 100
    endurance_actuelle: int = 100
    
    # Attributs principaux (1-100)
    force: int = 10
    vitesse: int = 10
    endurance: int = 10
    magie: int = 10
    perception: int = 10
    charisme: int = 10
    chance: int = 10
    intelligence: int = 10
    volonte: int = 10
    
    # Systèmes sociaux et psychologiques
    traits: TraitsPersonnalite = field(default_factory=TraitsPersonnalite)
    relations: Dict[str, Relation] = field(default_factory=dict)
    reputation: Dict[str, int] = field(default_factory=lambda: {})  # Réputation par faction/région
    
    # Systèmes de progression
    competences: Dict[str, Competence] = field(default_factory=dict)
    magies: Dict[str, CompetenceMagique] = field(default_factory=dict)
    equipement: Dict[str, Any] = field(default_factory=dict)
    inventaire: List[Dict[str, Any]] = field(default_factory=list)
    or_possede: int = 0
    
    # Variables pour les systèmes avancés
    affinites_elementaires: Dict[str, float] = field(default_factory=lambda: {})
    position_actuelle: Dict[str, Any] = field(default_factory=lambda: {"region": "", "lieu": "", "coordonnees": (0, 0)})
    quetes_actives: List[str] = field(default_factory=list)
    quetes_terminees: List[str] = field(default_factory=list)
    effets_statut: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Cycle de vie et évolution
    derniere_evolution_age: int = 0
    etapes_vie: List[str] = field(default_factory=list)
    objectifs_personnels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Événements importants
    historique_evenements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pour les PNJ canoniques de Mushoku Tensei
    est_canonique: bool = False
    apparitions_chronologiques: Dict[str, str] = field(default_factory=dict)  # Période -> Lieu
    dialogues_specifiques: Dict[str, List[str]] = field(default_factory=lambda: {})
    evenements_preprogrammes: List[Dict[str, Any]] = field(default_factory=list)

    def vieillir(self, annees: int = 1):
        """Fait vieillir le personnage et applique les changements appropriés"""
        ancien_age = self.age
        self.age += annees
        
        # Vérifier les étapes de vie
        if (ancien_age < 13 and self.age >= 13):
            self.etapes_vie.append("Adolescence")
        elif (ancien_age < 18 and self.age >= 18):
            self.etapes_vie.append("Âge adulte")
        elif (ancien_age < 40 and self.age >= 40):
            self.etapes_vie.append("Maturité")
        elif (ancien_age < 60 and self.age >= 60):
            self.etapes_vie.append("Vieillesse")
            
        # Ajustement des statistiques basé sur l'âge
        self.derniere_evolution_age = self.age
        
        # Enregistrer l'événement
        self.historique_evenements.append({
            "type": "vieillissement",
            "age": self.age,
            "changements": f"A vieilli de {annees} an(s)",
            "date": datetime.datetime.now().isoformat()
        })
        
    def ajouter_competence(self, nom: str, niveau: int = 1, description: str = ""):
        """Ajoute une nouvelle compétence au personnage"""
        if nom not in self.competences:
            self.competences[nom] = Competence(
                nom=nom,
                niveau=niveau,
                description=description
            )
            return True
        return False
        
    def ajouter_magie(self, nom: str, element: str, rang: str = "Débutant", cout: int = 5):
        """Ajoute une nouvelle magie au personnage"""
        if nom not in self.magies:
            self.magies[nom] = CompetenceMagique(
                nom=nom,
                element=element,
                rang=rang,
                cout_mana=cout
            )
            return True
        return False

# ===================================================
# === BLOC 3/12 : SYSTÈME DU MONDE ET ENVIRONNEMENT ===
# ===================================================

# Structure de date dans l'univers du jeu
@dataclass
class DateMonde:
    annee: int = 407  # Année de début selon la chronologie de Mushoku Tensei
    mois: int = 1     # 1-12
    jour: int = 1     # 1-30 (chaque mois a 30 jours pour simplifier)
    heure: int = 12
    minute: int = 0
    
    def avancer_temps(self, jours: int = 0, heures: int = 0, minutes: int = 0):
        """Avance le temps du monde d'un nombre spécifié de jours, heures et minutes"""
        total_minutes = self.minute + minutes
        heures_ajoutees = total_minutes // 60
        self.minute = total_minutes % 60
        
        total_heures = self.heure + heures + heures_ajoutees
        jours_ajoutes = total_heures // 24
        self.heure = total_heures % 24
        
        total_jours = self.jour + jours + jours_ajoutes
        mois_ajoutes = (total_jours - 1) // 30
        self.jour = ((total_jours - 1) % 30) + 1
        
        total_mois = self.mois + mois_ajoutes
        annees_ajoutees = (total_mois - 1) // 12
        self.mois = ((total_mois - 1) % 12) + 1
        
        self.annee += annees_ajoutees
    
    def saison_actuelle(self) -> str:
        """Retourne la saison actuelle basée sur le mois"""
        return SAISONS[self.mois - 1]
    
    def __str__(self) -> str:
        """Représentation textuelle de la date"""
        return f"{self.jour} {MOIS[self.mois-1]} {self.annee}, {self.heure:02d}:{self.minute:02d}"

# Structure pour les phénomènes astronomiques et cosmiques
@dataclass
class PhenomeneCeleste:
    nom: str
    description: str
    date_debut: DateMonde
    duree_jours: int
    effets: Dict[str, Any]
    rarete: int = 1  # 1-10, 10 étant le plus rare
    visible: bool = True
    
    def est_actif(self, date_actuelle: DateMonde) -> bool:
        """Vérifie si le phénomène est actif à une date donnée"""
        # Conversion en jours absolus pour simplifier la comparaison
        def date_en_jours(date: DateMonde) -> int:
            return date.annee * 360 + (date.mois - 1) * 30 + date.jour
        
        debut_jours = date_en_jours(self.date_debut)
        date_jours = date_en_jours(date_actuelle)
        return debut_jours <= date_jours <= (debut_jours + self.duree_jours)

# Classes pour la météo et les conditions environnementales
class ConditionMeteo(Enum):
    ENSOLEILLE = auto()
    NUAGEUX = auto()
    PLUIE = auto()
    ORAGE = auto()
    NEIGE = auto()
    TEMPETE = auto()
    BRUME = auto()
    CANICULE = auto()

@dataclass
class MeteoRegionale:
    condition: ConditionMeteo = ConditionMeteo.ENSOLEILLE
    intensite: int = 1  # 1-10
    temperature: int = 20  # en degrés Celsius
    humidite: int = 50  # pourcentage
    vent: int = 0  # vitesse en km/h
    duree: int = 24  # durée en heures
    effets_speciaux: Dict[str, Any] = field(default_factory=dict)

    def affecter_combat(self) -> Dict[str, int]:
        """Retourne les modificateurs à appliquer au combat basés sur la météo"""
        modificateurs = {}
        
        if self.condition == ConditionMeteo.PLUIE:
            modificateurs["precision"] = -5 * self.intensite // 3
            modificateurs["feu"] = -10 * self.intensite // 2
            modificateurs["eau"] = 5 * self.intensite // 2
            
        elif self.condition == ConditionMeteo.NEIGE:
            modificateurs["vitesse"] = -5 * self.intensite // 2
            modificateurs["glace"] = 7 * self.intensite // 2
            
        # Plus de conditions et d'effets...
            
        return modificateurs

    def affecter_magie(self, element: str) -> float:
        """Retourne le multiplicateur à appliquer à un élément magique"""
        multiplicateur = 1.0
        
        if self.condition == ConditionMeteo.ORAGE and element == "Foudre":
            multiplicateur += 0.2 * self.intensite
        elif self.condition == ConditionMeteo.CANICULE and element == "Feu":
            multiplicateur += 0.15 * self.intensite
        # Plus de conditions...
            
        return multiplicateur
        
# Structure pour les emplacements et lieux
@dataclass
class Emplacement:
    id: str
    nom: str
    type: str  # ville, donjon, forêt, etc.
    region: str
    continent: str
    description: str = ""
    danger: int = 0  # Niveau de danger (0-10)
    coordonnees: Tuple[float, float] = (0.0, 0.0)
    connecte_a: List[str] = field(default_factory=list)  # IDs des emplacements connectés
    services: Dict[str, bool] = field(default_factory=lambda: {})  # auberge, forge, etc.
    habitants: List[str] = field(default_factory=list)  # IDs des PNJs présents
    quetes_disponibles: List[str] = field(default_factory=list)
    ressources: Dict[str, int] = field(default_factory=dict)  # ressources disponibles
    
    # Nouvelles propriétés pour les systèmes dynamiques
    meteo_actuelle: MeteoRegionale = field(default_factory=MeteoRegionale)
    evenements_actifs: List[str] = field(default_factory=list)
    restrictions_magiques: Dict[str, float] = field(default_factory=dict)  # Modificateurs par élément
    
    # Pour le système d'écosystème dynamique
    faune_locale: Dict[str, int] = field(default_factory=dict)  # Espèce -> Population
    flore_locale: Dict[str, int] = field(default_factory=dict)  # Type -> Abondance
    equilibre_ecologique: int = 5  # 0-10, 5 étant l'équilibre normal
    
    def mettre_a_jour_meteo(self, date_monde: DateMonde):
        """Met à jour la météo en fonction de la date, saison, et autres facteurs"""
        saison = date_monde.saison_actuelle()
        
        # Chances de base pour chaque condition selon la saison
        chances = {
            "Hiver": {
                ConditionMeteo.ENSOLEILLE: 20, 
                ConditionMeteo.NUAGEUX: 30,
                ConditionMeteo.NEIGE: 40,
                ConditionMeteo.TEMPETE: 10
            },
            "Printemps": {
                ConditionMeteo.ENSOLEILLE: 40, 
                ConditionMeteo.NUAGEUX: 30,
                ConditionMeteo.PLUIE: 25,
                ConditionMeteo.ORAGE: 5
            },
            # Plus de saisons...
        }
        
        # Sélection aléatoire basée sur les chances
        if saison in chances:
            conditions = list(chances[saison].keys())
            poids = list(chances[saison].values())
            condition_choisie = random.choices(conditions, weights=poids, k=1)[0]
            
            # Déterminer l'intensité
            intensite = random.randint(1, 5)  # Base
            if random.random() < 0.1:  # 10% de chance d'événement extrême
                intensite += random.randint(3, 5)
                
            # Créer la nouvelle météo
            self.meteo_actuelle = MeteoRegionale(
                condition=condition_choisie,
                intensite=min(10, intensite),
                temperature=self._calculer_temperature(saison, condition_choisie),
                duree=random.randint(12, 48)  # Durée entre 12h et 2 jours
            )
            
    def _calculer_temperature(self, saison: str, condition: ConditionMeteo) -> int:
        """Calcule la température en fonction de la saison et des conditions météo"""
        base_temp = {
            "Hiver": -5, 
            "Printemps": 15, 
            "Été": 25, 
            "Automne": 10
        }.get(saison, 15)
        
        modificateurs = {
            ConditionMeteo.ENSOLEILLE: 5,
            ConditionMeteo.NUAGEUX: 0,
            ConditionMeteo.PLUIE: -3,
            ConditionMeteo.NEIGE: -10,
            ConditionMeteo.TEMPETE: -5,
            ConditionMeteo.CANICULE: 15
        }
        
        return base_temp + modificateurs.get(condition, 0) + random.randint(-2, 2)

# Système de factions et d'organisations
@dataclass
class Faction:
    id: str
    nom: str
    description: str = ""
    influence: int = 0  # 0-100
    alignement: Alignement = Alignement.NEUTRE
    territoires: List[str] = field(default_factory=list)
    membres_importants: List[str] = field(default_factory=list)
    ressources: Dict[str, int] = field(default_factory=dict)
    relations: Dict[str, int] = field(default_factory=dict)  # ID faction -> Valeur relation
    
    # Pour système politique dynamique
    objectifs: List[Dict[str, Any]] = field(default_factory=list)
    evenements_historiques: List[Dict[str, Any]] = field(default_factory=list)
    
    def modifier_relation(self, faction_id: str, valeur: int):
        """Modifie la relation avec une autre faction"""
        relation_actuelle = self.relations.get(faction_id, 0)
        nouvelle_relation = max(-100, min(100, relation_actuelle + valeur))
        self.relations[faction_id] = nouvelle_relation

# ===========================================================
# === BLOC 4/12 : SYSTÈME DE QUÊTES ET ÉVÉNEMENTS ===
# ===========================================================

# Enums pour les quêtes et événements
class TypeQuete(Enum):
    PRINCIPALE = auto()
    SECONDAIRE = auto()
    FACTION = auto()
    PERSONNELLE = auto()
    QUOTIDIENNE = auto()
    MONDE = auto()

class DifficulteQuete(Enum):
    TRIVIALE = auto()
    FACILE = auto()
    MOYENNE = auto()
    DIFFICILE = auto()
    TRES_DIFFICILE = auto()
    LEGENDAIRE = auto()
    
class StatutQuete(Enum):
    DISPONIBLE = auto()
    ACTIVE = auto()
    COMPLETEE = auto()
    ECHOUEE = auto()
    EXPIREE = auto()
    CACHEE = auto()

# Système de quêtes détaillé
@dataclass
class ObjectifQuete:
    description: str
    quantite_cible: int = 1
    progres_actuel: int = 0
    complete: bool = False
    cible_id: Optional[str] = None
    emplacement_id: Optional[str] = None
    type_objectif: str = "standard"  # standard, collecte, assassinat, escorte, etc.

@dataclass
class Quete:
    id: str
    titre: str
    description: str
    type: TypeQuete = TypeQuete.SECONDAIRE
    difficulte: DifficulteQuete = DifficulteQuete.MOYENNE
    statut: StatutQuete = StatutQuete.DISPONIBLE
    objectifs: List[ObjectifQuete] = field(default_factory=list)
    
    donneur_id: Optional[str] = None
    recompenses: Dict[str, Any] = field(default_factory=dict)
    prerequis: Dict[str, Any] = field(default_factory=dict)
    delai_jours: Optional[int] = None
    date_debut: Optional[DateMonde] = None
    date_expiration: Optional[DateMonde] = None
    
    # Pour le système de quêtes dynamiques
    consequences: List[Dict[str, Any]] = field(default_factory=list)
    chemins_alternatifs: Dict[str, List[ObjectifQuete]] = field(default_factory=dict)
    
    # Pour quêtes liées à l'histoire canonique
    est_canonique: bool = False
    impact_monde: int = 0  # 0-10, indique à quel point cette quête affecte le monde
    
    def est_complete(self) -> bool:
        """Vérifie si tous les objectifs sont complétés"""
        if not self.objectifs:
            return False
        return all(obj.complete for obj in self.objectifs)
    
    def est_expiree(self, date_actuelle: DateMonde) -> bool:
        """Vérifie si la quête a expiré"""
        if self.date_expiration is None:
            return False
            
        # Conversion en jours pour simplifier la comparaison
        def en_jours(date: DateMonde) -> int:
            return date.annee * 360 + (date.mois - 1) * 30 + date.jour
            
        return en_jours(date_actuelle) > en_jours(self.date_expiration)
    
    def mettre_a_jour_statut(self, date_actuelle: DateMonde):
        """Met à jour le statut de la quête en fonction de sa progression et de la date"""
        if self.statut == StatutQuete.ACTIVE:
            if self.est_complete():
                self.statut = StatutQuete.COMPLETEE
            elif self.est_expiree(date_actuelle):
                self.statut = StatutQuete.ECHOUEE

# Système d'événements du monde
class TypeEvenement(Enum):
    FESTIVAL = auto()
    CATASTROPHE = auto()
    GUERRE = auto()
    POLITIQUE = auto()
    METEO = auto()
    RITUEL = auto()
    COSMIQUE = auto()
    PERSONNEL = auto()

@dataclass
class Evenement:
    id: str
    titre: str
    description: str
    type: TypeEvenement
    importance: int = 1  # 1-10
    regions_affectees: List[str] = field(default_factory=list)
    date_debut: DateMonde = field(default_factory=DateMonde)
    duree_jours: int = 1
    
    # Effets sur le monde et personnages
    modificateurs_mondiaux: Dict[str, Any] = field(default_factory=dict)
    effets_personnage: Dict[str, Any] = field(default_factory=dict)
    
    # Pour événements dynamiques et cascades d'effets
    declencheurs: List[Dict[str, Any]] = field(default_factory=list)
    consequences_possibles: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pour événements liés à l'histoire canonique
    est_canonique: bool = False
    prevention_possible: bool = True
    
    def est_actif(self, date_actuelle: DateMonde) -> bool:
        """Vérifie si l'événement est actif à une date donnée"""
        # Conversion en jours pour simplifier la comparaison
        def en_jours(date: DateMonde) -> int:
            return date.annee * 360 + (date.mois - 1) * 30 + date.jour
        
        debut = en_jours(self.date_debut)
        maintenant = en_jours(date_actuelle)
        fin = debut + self.duree_jours
        
        return debut <= maintenant <= fin

# Système de rêves et visions
@dataclass
class Reve:
    id: str
    titre: str
    contenu: str
    symbolisme: Dict[str, str] = field(default_factory=dict)
    est_prophetique: bool = False
    lien_memoire: Optional[str] = None  # Lien vers un souvenir ou événement
    
    # Pour les rêves prophétiques
    probabilite_realisation: int = 0  # 0-100
    evenement_predit_id: Optional[str] = None
    
    # Pour l'interprétation
    indices: List[str] = field(default_factory=list)
    interpretations_possibles: List[str] = field(default_factory=list)

# Gestionnaire de synchronicité narrative
class GestionnaireSynchronicite:
    def __init__(self):
        self.evenements_recents = deque(maxlen=20)
        self.motifs_detectes = []
        self.etat_synchronicite = 0  # -10 à 10, négatif = chaos, positif = destin/harmonie
    
    def enregistrer_evenement(self, type_evenement: str, details: Dict[str, Any]):
        """Enregistre un nouvel événement dans la séquence"""
        self.evenements_recents.append({
            "type": type_evenement,
            "details": details,
            "date": datetime.datetime.now().isoformat()
        })
        self._analyser_motifs()
    
    def _analyser_motifs(self):
        """Analyse les motifs dans les événements récents"""
        # Logique simplifiée pour détecter les modèles récurrents ou significatifs
        # Une implémentation réelle utiliserait des algorithmes plus sophistiqués
        
        # Exemple: détection de répétitions
        types_evenements = [e["type"] for e in self.evenements_recents]
        repetitions = {t: types_evenements.count(t) for t in set(types_evenements)}
        
        # Chercher des séquences particulières
        for type_ev, count in repetitions.items():
            if count >= 3:  # Si un type d'événement se répète 3 fois ou plus
                motif = f"Répétition de '{type_ev}' ({count} occurrences)"
                if motif not in self.motifs_detectes:
                    self.motifs_detectes.append(motif)
                    # Ajuster le niveau de synchronicité
                    self.etat_synchronicite += 1
    
    def generer_evenement_synchronique(self) -> Optional[Dict[str, Any]]:
        """Génère un événement basé sur la synchronicité narrative actuelle"""
        if abs(self.etat_synchronicite) < 3:
            # Pas assez de synchronicité pour générer un événement significatif
            return None
            
        # Plus le niveau est élevé (positif ou négatif), plus l'événement est important
        importance = min(10, abs(self.etat_synchronicite))
        
        if self.etat_synchronicite > 0:
            # Synchronicité harmonieuse - événements "destinés"
            return {
                "type": "synchronicite_positive",
                "importance": importance,
                "description": "Un événement étrangement opportun se produit...",
                "nature": random.choice(["rencontre_fortuite", "découverte_inattendue", "coïncidence_favorable"])
            }
        else:
            # Synchronicité chaotique - événements perturbateurs
            return {
                "type": "synchronicite_negative",
                "importance": importance,
                "description": "Un événement troublant brise l'harmonie...",
                "nature": random.choice(["obstacle_inattendu", "révélation_perturbante", "retournement_destin"])
            }

# ===========================================================
# === BLOC 5/12 : SYSTÈME DE MAGIE ET COMBAT ===
# ===========================================================

# Enums pour le système de combat et magie
class TypeAttaque(Enum):
    MELEE = auto()
    DISTANCE = auto()
    MAGIE = auto()
    FURTIF = auto()

class StyleCombat(Enum):
    OFFENSIF = auto()
    DEFENSIF = auto()
    EQUILIBRE = auto()
    TECHNIQUE = auto()
    BERSERKER = auto()

class Portee(Enum):
    CONTACT = auto()
    COURTE = auto()
    MOYENNE = auto()
    LONGUE = auto()
    TRES_LONGUE = auto()

# Système magique de Mushoku Tensei
@dataclass
class Sort:
    id: str
    nom: str
    element: str  # Eau, Feu, Terre, Vent, etc.
    rang: str  # Débutant, Intermédiaire, Avancé, Saint, Roi, Empereur, Dieu
    cout_mana: int
    temps_incantation: float  # en secondes
    description: str = ""
    effets: Dict[str, Any] = field(default_factory=dict)
    portee: Portee = Portee.MOYENNE
    zone_effet: int = 0  # 0 = cible unique, >0 = rayon en mètres
    
    # Pour le système d'évolution magique
    style_utilisation: Dict[str, int] = field(default_factory=lambda: {
        "offensif": 0, "défensif": 0, "utilitaire": 0, "soutien": 0
    })
    evolution_possible: Dict[str, str] = field(default_factory=dict)  # Direction -> Nouveau sort
    conditions_evolution: Dict[str, int] = field(default_factory=dict)
    
    # Effets visuels et sensoriels
    manifestation: str = ""  # Description de l'apparence du sort
    
    def calculer_puissance(self, lanceur: 'Personnage', modificateurs: Dict[str, Any] = None) -> int:
        """Calcule la puissance du sort en fonction du lanceur et des modificateurs"""
        if modificateurs is None:
            modificateurs = {}
            
        # Base de puissance selon le rang
        base_puissance = {
            "Débutant": 10,
            "Intermédiaire": 25,
            "Avancé": 50,
            "Saint": 100,
            "Roi": 250,
            "Empereur": 500,
            "Dieu": 1000
        }.get(self.rang, 10)
        
        # Facteurs du lanceur
        niveau_magie = lanceur.magie
        affinite = lanceur.affinites_elementaires.get(self.element, 1.0)
        
        # Calculer la puissance finale
        puissance = base_puissance * (1 + niveau_magie/100) * affinite
        
        # Appliquer les modificateurs
        for mod, valeur in modificateurs.items():
            if mod == "meteo" and isinstance(valeur, float):
                puissance *= valeur
            elif mod == "concentration" and isinstance(valeur, int):
                puissance *= (1 + valeur/100)
            # Plus de modificateurs...
                
        return int(puissance)
    
    def verifier_evolution(self) -> Optional[str]:
        """Vérifie si le sort peut évoluer et retourne la direction d'évolution"""
        for direction, seuil in self.conditions_evolution.items():
            if self.style_utilisation.get(direction, 0) >= seuil:
                return direction
        return None
        
    def enregistrer_utilisation(self, style: str):
        """Enregistre une utilisation du sort dans un style particulier"""
        if style in self.style_utilisation:
            self.style_utilisation[style] += 1

# Système d'arts martiaux et d'épée
@dataclass
class Technique:
    id: str
    nom: str
    ecole: str  # Style Eau, Style Nord, Style Dieu-Épée, etc.
    rang: int  # 1-10
    type: TypeAttaque = TypeAttaque.MELEE
    cout_endurance: int = 5
    prerequis: Dict[str, int] = field(default_factory=dict)
    effets: Dict[str, Any] = field(default_factory=dict)
    
    # Pour progression et adaptation
    adaptations_possibles: List[Dict[str, Any]] = field(default_factory=list)
    maitrise: int = 0  # 0-100
    
    def calculer_puissance(self, utilisateur: 'Personnage', arme: Optional['Objet'] = None) -> int:
        """Calcule la puissance de la technique basée sur l'utilisateur et son arme"""
        # Facteurs de base
        puissance_base = self.rang * 15
        
        # Facteurs du personnage
        niveau_force = utilisateur.force
        niveau_vitesse = utilisateur.vitesse
        
        # Facteurs de l'arme
        bonus_arme = 0
        if arme:
            bonus_arme = arme.puissance
            
        # Bonus de maîtrise
        bonus_maitrise = self.maitrise / 100
        
        # Calculer la puissance
        puissance = puissance_base * (1 + niveau_force/100) * (1 + niveau_vitesse/200)
        puissance += bonus_arme * (1 + bonus_maitrise)
        
        return int(puissance)

# Système d'effets de statut
@dataclass
class EffetStatut:
    id: str
    nom: str
    description: str
    duree: int  # en tours ou en durée temporelle
    type: str  # positif, négatif, neutre
    cumul: bool = False  # si l'effet peut se cumuler
    effets: Dict[str, Any] = field(default_factory=dict)
    
    def appliquer_effet(self, cible: 'Personnage'):
        """Applique l'effet du statut à la cible"""
        for attribut, modification in self.effets.items():
            if hasattr(cible, attribut):
                valeur_actuelle = getattr(cible, attribut)
                
                # Application différente selon le type de modification
                if isinstance(modification, int) or isinstance(modification, float):
                    nouvelle_valeur = valeur_actuelle + modification
                elif isinstance(modification, dict) and "multiplicateur" in modification:
                    nouvelle_valeur = valeur_actuelle * modification["multiplicateur"]
                else:
                    continue
                    
                # Appliquer la nouvelle valeur
                setattr(cible, attribut, nouvelle_valeur)
                
    def retirer_effet(self, cible: 'Personnage'):
        """Retire l'effet du statut de la cible"""
        for attribut, modification in self.effets.items():
            if hasattr(cible, attribut):
                valeur_actuelle = getattr(cible, attribut)
                
                # Retrait différent selon le type de modification
                if isinstance(modification, int) or isinstance(modification, float):
                    nouvelle_valeur = valeur_actuelle - modification
                elif isinstance(modification, dict) and "multiplicateur" in modification:
                    nouvelle_valeur = valeur_actuelle / modification["multiplicateur"]
                else:
                    continue
                    
                # Appliquer la nouvelle valeur
                setattr(cible, attribut, nouvelle_valeur)

# Système de combat
@dataclass
class ResultatAction:
    succes: bool
    message: str
    dommages: int = 0
    effets_secondaires: List[Dict[str, Any]] = field(default_factory=list)
    modificateurs_stats: Dict[str, Any] = field(default_factory=dict)

class GestionnaireCombat:
    def __init__(self):
        self.participants = []
        self.tour_actuel = 0
        self.actions_tour = []
        self.log_combat = []
        self.statut_combat = "en_préparation"  # en_préparation, actif, terminé
        self.modificateurs_environnement = {}
        self.effets_actifs = {}
    
    def ajouter_participant(self, personnage: 'Personnage', equipe: int = 0):
        """Ajoute un participant au combat"""
        self.participants.append({
            "personnage": personnage,
            "equipe": equipe,
            "initiative": self._calculer_initiative(personnage),
            "effets_actifs": {},
            "actions_disponibles": True,
            "est_hors_combat": False
        })
    
    def _calculer_initiative(self, personnage: 'Personnage') -> int:
        """Calcule l'initiative d'un personnage pour déterminer l'ordre d'action"""
        base = personnage.vitesse
        bonus_random = random.randint(1, 20)
        return base + bonus_random
    
    def demarrer_combat(self):
        """Démarre le combat et trie les participants selon leur initiative"""
        if not self.participants:
            return False
            
        # Trier par initiative
        self.participants.sort(key=lambda p: p["initiative"], reverse=True)
        
        self.statut_combat = "actif"
        self.tour_actuel = 1
        
        # Enregistrer le début du combat
        self.log_combat.append({
            "tour": 0,
            "type": "debut_combat",
            "participants": [
                {"nom": p["personnage"].nom, "equipe": p["equipe"], "initiative": p["initiative"]} 
                for p in self.participants
            ]
        })
        
        return True
    
    def action_combat(self, index_attaquant: int, type_action: str, 
                     cibles: List[int], details_action: Dict[str, Any]) -> ResultatAction:
        """Traite une action de combat d'un participant"""
        if self.statut_combat != "actif" or index_attaquant >= len(self.participants):
            return ResultatAction(False, "Action impossible - combat non actif ou participant invalide")
            
        attaquant = self.participants[index_attaquant]
        if attaquant["est_hors_combat"] or not attaquant["actions_disponibles"]:
            return ResultatAction(False, "Ce personnage ne peut pas agir pour le moment")
        
        # Traitement selon le type d'action
        if type_action == "attaque_normale":
            return self._traiter_attaque_normale(attaquant, cibles, details_action)
        elif type_action == "technique":
            return self._traiter_technique(attaquant, cibles, details_action)
        elif type_action == "sort":
            return self._traiter_sort(attaquant, cibles, details_action)
        elif type_action == "objet":
            return self._traiter_objet(attaquant, cibles, details_action)
        elif type_action == "defense":
            return self._traiter_defense(attaquant)
        else:
            return ResultatAction(False, f"Type d'action non reconnu: {type_action}")
    
    def _traiter_attaque_normale(self, attaquant, cibles, details):
        """Traite une attaque normale"""
        resultats = []
        personnage_attaquant = attaquant["personnage"]
        
        for index_cible in cibles:
            if index_cible >= len(self.participants):
                resultats.append(ResultatAction(False, "Cible invalide"))
                continue
                
            cible = self.participants[index_cible]
            if cible["est_hors_combat"]:
                resultats.append(ResultatAction(False, f"{cible['personnage'].nom} est déjà hors combat"))
                continue
                
            # Calcul de l'attaque
            force_attaque = personnage_attaquant.force
            if "arme" in details:
                force_attaque += details["arme"].get("puissance", 0)
                
            # Calcul de la défense
            defense_cible = cible["personnage"].endurance
            
            # Calcul des dégâts
            degats_base = max(1, force_attaque - defense_cible // 2)
            variation = random.uniform(0.8, 1.2)
            degats = int(degats_base * variation)
            
            # Application des dégâts
            cible["personnage"].sante_actuelle -= degats
            
            # Vérifier si hors combat
            if cible["personnage"].sante_actuelle <= 0:
                cible["personnage"].sante_actuelle = 0
                cible["est_hors_combat"] = True
                message = f"{personnage_attaquant.nom} attaque {cible['personnage'].nom} pour {degats} dégâts et le met hors combat!"
            else:
                message = f"{personnage_attaquant.nom} attaque {cible['personnage'].nom} pour {degats} dégâts."
                
            resultats.append(ResultatAction(True, message, degats))
            
            # Enregistrer l'action dans le log
            self.log_combat.append({
                "tour": self.tour_actuel,
                "type": "attaque_normale",
                "attaquant": personnage_attaquant.nom,
                "cible": cible["personnage"].nom,
                "degats": degats,
                "hors_combat": cible["est_hors_combat"]
            })
            
        # Marquer l'attaquant comme ayant agi
        attaquant["actions_disponibles"] = False
        
        # Retourner le résultat composite ou le premier résultat
        if resultats:
            return resultats[0]  # Simplification - on pourrait retourner un résultat composite
        else:
            return ResultatAction(False, "Aucune action effectuée")
            
    # Autres méthodes pour traiter les différents types d'actions...
    def _traiter_technique(self, attaquant, cibles, details):
        """Implémentation de l'utilisation des techniques de combat"""
        pass
        
    def _traiter_sort(self, attaquant, cibles, details):
        """Implémentation du lancement de sorts"""
        pass
        
    def _traiter_objet(self, attaquant, cibles, details):
        """Implémentation de l'utilisation d'objets"""
        pass
        
    def _traiter_defense(self, attaquant):
        """Implémentation de l'action de défense"""
        pass
        
    def tour_suivant(self):
        """Passe au tour suivant de combat"""
        if self.statut_combat != "actif":
            return False
            
        self.tour_actuel += 1
        
        # Réinitialiser les actions disponibles
        for participant in self.participants:
            if not participant["est_hors_combat"]:
                participant["actions_disponibles"] = True
                
        # Appliquer les effets de statut actifs
        self._appliquer_effets_statut()
        
        # Vérifier si le combat est terminé
        self._verifier_fin_combat()
        
        # Enregistrer le début du nouveau tour
        self.log_combat.append({
            "tour": self.tour_actuel,
            "type": "debut_tour"
        })
        
        return True
        
    def _appliquer_effets_statut(self):
        """Applique les effets de statut actifs et réduit leur durée"""
        pass
        
    def _verifier_fin_combat(self):
        """Vérifie si le combat est terminé (une équipe éliminée ou en fuite)"""
        pass

# ==========================================================
# === BLOC 6/12 : SYSTÈME D'OBJETS ET D'INVENTAIRE ===
# ==========================================================

# Enums pour les objets et l'inventaire
class RareteObjet(Enum):
    COMMUN = auto()
    RARE = auto()
    EPIQUE = auto()
    LEGENDAIRE = auto()
    DIVIN = auto()
    UNIQUE = auto()

class TypeObjet(Enum):
    ARME = auto()
    ARMURE = auto()
    ACCESSOIRE = auto()
    CONSOMMABLE = auto()
    RESSOURCE = auto()
    LIVRE = auto()
    QUETE = auto()
    MAGIQUE = auto()
    SPECIAL = auto()

# Structure de base pour tous les objets
@dataclass
class Objet:
    id: str
    nom: str
    description: str
    type: TypeObjet
    rarete: RareteObjet = RareteObjet.COMMUN
    poids: float = 0.0  # en kg
    valeur: int = 0  # valeur marchande en pièces
    est_empiable: bool = False
    quantite: int = 1
    durabilite_max: int = 100
    durabilite: int = 100
    image: Optional[str] = None  # chemin vers une image représentative
    
    # Métadonnées
    region_origine: Optional[str] = None
    fabricant: Optional[str] = None
    age_objet: Optional[int] = None
    lore: Optional[str] = None
    
    # Fonctions d'utilisation
    utilisable: bool = False
    equipable: bool = False
    consommable: bool = False
    
    # Pour le système d'objets spéciaux
    effets_speciaux: Dict[str, Any] = field(default_factory=dict)
    mots_activation: List[str] = field(default_factory=list)
    
    def utiliser(self, utilisateur: 'Personnage') -> Dict[str, Any]:
        """Utilise l'objet et retourne le résultat de l'utilisation"""
        if not self.utilisable:
            return {"succes": False, "message": f"{self.nom} n'est pas utilisable."}
            
        resultat = {"succes": True, "message": f"{utilisateur.nom} utilise {self.nom}."}
        
        # Si c'est un consommable, diminuer la quantité
        if self.consommable:
            if self.quantite <= 0:
                return {"succes": False, "message": f"Plus de {self.nom} disponible."}
            self.quantite -= 1
            
        # Appliquer les effets spéciaux si présents
        for effet, valeur in self.effets_speciaux.items():
            if effet == "soin" and isinstance(valeur, int):
                soin = min(valeur, utilisateur.sante_max - utilisateur.sante_actuelle)
                utilisateur.sante_actuelle += soin
                resultat["message"] += f" Récupère {soin} points de santé."
            elif effet == "mana" and isinstance(valeur, int):
                mana = min(valeur, utilisateur.mana_max - utilisateur.mana_actuel)
                utilisateur.mana_actuel += mana
                resultat["message"] += f" Récupère {mana} points de mana."
            # Autres effets possibles...
                
        return resultat
    
    def peut_empiler_avec(self, autre: 'Objet') -> bool:
        """Vérifie si cet objet peut être empilé avec un autre"""
        if not self.est_empiable or not autre.est_empiable:
            return False
            
        # Les objets doivent être identiques pour être empilés
        return (self.id == autre.id and 
                self.nom == autre.nom and 
                self.rarete == autre.rarete and
                self.type == autre.type)

# Types d'objets spécifiques
@dataclass
class Arme(Objet):
    type: TypeObjet = TypeObjet.ARME
    equipable: bool = True
    
    # Attributs spécifiques aux armes
    puissance: int = 0
    precision: int = 0
    vitesse: int = 0
    portee: Portee = Portee.CONTACT
    style_compatible: List[str] = field(default_factory=list)
    
    # Pour le système d'évolution des armes
    experience: int = 0
    niveau: int = 1
    capacites_debloquees: List[str] = field(default_factory=list)
    
    def calculer_degats(self, utilisateur: 'Personnage') -> Tuple[int, bool]:
        """Calcule les dégâts de l'arme en fonction de l'utilisateur et retourne un tuple (dégâts, critique)"""
        degats_base = self.puissance + (utilisateur.force // 5)
        multiplicateur = 1.0
        
        # Bonus de niveau
        multiplicateur += (self.niveau - 1) * 0.1
        
        # Vérifier coup critique
        est_critique = random.random() < (utilisateur.chance / 500)
        if est_critique:
            multiplicateur *= 1.5
            
        # Calcul final
        degats = int(degats_base * multiplicateur)
        
        return (degats, est_critique)

@dataclass
class Armure(Objet):
    type: TypeObjet = TypeObjet.ARMURE
    equipable: bool = True
    
    # Attributs spécifiques aux armures
    defense: int = 0
    resistance_magique: int = 0
    mobilite: int = 0  # Impact sur la vitesse/agilité
    emplacement: str = "corps"  # corps, tête, jambes, etc.
    
    # Résistances élémentaires (0-100%)
    resistances: Dict[str, int] = field(default_factory=lambda: {
        "physique": 0, "feu": 0, "eau": 0, "terre": 0, "vent": 0, 
        "foudre": 0, "glace": 0, "lumière": 0, "ombre": 0
    })

@dataclass
class Consommable(Objet):
    type: TypeObjet = TypeObjet.CONSOMMABLE
    utilisable: bool = True
    consommable: bool = True
    
    # Attributs spécifiques aux consommables
    effets_instant: Dict[str, int] = field(default_factory=dict)
    effets_duree: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duree_effets: int = 0  # en tours ou en minutes/heures selon contexte

@dataclass
class ArtefactMagique(Objet):
    type: TypeObjet = TypeObjet.MAGIQUE
    rarete: RareteObjet = RareteObjet.RARE
    
    # Attributs spécifiques aux artefacts
    puissance_magique: int = 0
    element_associe: Optional[str] = None
    charges: Optional[int] = None
    
    # Pour artefacts spéciaux liés à l'univers de Mushoku Tensei
    est_relique: bool = False
    histoire_relique: str = ""
    personnage_canonique_lie: Optional[str] = None

# Système d'inventaire
@dataclass
class Inventaire:
    capacite_max: int = 50  # Nombre d'emplacements
    poids_max: float = 100.0  # en kg
    objets: Dict[str, Objet] = field(default_factory=dict)
    or_possede: int = 0
    equipement_actif: Dict[str, str] = field(default_factory=dict)  # emplacement -> id objet
    
    def calculer_poids_total(self) -> float:
        """Calcule le poids total de l'inventaire"""
        return sum(obj.poids * obj.quantite for obj in self.objets.values())
        
    def espace_restant(self) -> int:
        """Retourne le nombre d'emplacements restants"""
        return self.capacite_max - len(self.objets)
        
    def ajouter_objet(self, objet: Objet) -> bool:
        """Ajoute un objet à l'inventaire, en l'empilant si possible"""
        # Vérifier si l'inventaire est plein
        if len(self.objets) >= self.capacite_max and not any(
            objet.peut_empiler_avec(obj) for obj in self.objets.values()):
            return False
            
        # Vérifier le poids
        nouveau_poids = self.calculer_poids_total() + (objet.poids * objet.quantite)
        if nouveau_poids > self.poids_max:
            return False
            
        # Essayer d'empiler avec un objet existant
        for id_existant, obj_existant in self.objets.items():
            if objet.peut_empiler_avec(obj_existant):
                obj_existant.quantite += objet.quantite
                return True
                
        # Sinon, ajouter comme nouvel objet
        self.objets[objet.id] = objet
        return True
        
    def retirer_objet(self, objet_id: str, quantite: int = 1) -> Optional[Objet]:
        """Retire une certaine quantité d'un objet de l'inventaire"""
        if objet_id not in self.objets:
            return None
            
        objet = self.objets[objet_id]
        
        if objet.quantite <= quantite:
            # Retirer complètement l'objet
            del self.objets[objet_id]
            return objet
        else:
            # Réduire la quantité
            objet.quantite -= quantite
            
            # Créer un nouvel objet pour retourner la quantité demandée
            objet_retire = copy.deepcopy(objet)
            objet_retire.quantite = quantite
            return objet_retire
            
    def equiper(self, objet_id: str) -> Dict[str, Any]:
        """Équipe un objet et retourne le résultat de l'action"""
        if objet_id not in self.objets:
            return {"succes": False, "message": "Objet non trouvé dans l'inventaire."}
            
        objet = self.objets[objet_id]
        if not objet.equipable:
            return {"succes": False, "message": f"{objet.nom} n'est pas équipable."}
            
        # Déterminer l'emplacement
        if isinstance(objet, Armure):
            emplacement = objet.emplacement
        elif isinstance(objet, Arme):
            emplacement = "arme"
        else:
            emplacement = "accessoire"
            
        # Déséquiper l'objet actuel à cet emplacement
        if emplacement in self.equipement_actif:
            ancien_id = self.equipement_actif[emplacement]
            if ancien_id in self.objets:
                # Action spécifique au déséquipement si nécessaire
                pass
                
        # Équiper le nouvel objet
        self.equipement_actif[emplacement] = objet_id
            
        return {
            "succes": True,
            "message": f"{objet.nom} équipé(e) à l'emplacement {emplacement}."
        }

# Système de commerce et d'économie
class Marchand:
    def __init__(self, nom: str, specialite: str = "général"):
        self.nom = nom
        self.specialite = specialite
        self.inventaire = {}  # id -> (objet, prix_vente, prix_achat)
        self.capital = 1000  # Or disponible pour acheter aux joueurs
        self.marge = 1.5  # Multiplicateur de prix par rapport à la valeur de base
        self.relation_client = 50  # 0-100, influence les prix
        
    def ajouter_stock(self, objet: Objet, quantite: int = 1, prix_special: Optional[int] = None):
        """Ajoute un objet au stock du marchand"""
        if objet.id in self.inventaire:
            # Augmenter la quantité existante
            obj_stock, prix_vente, prix_achat = self.inventaire[objet.id]
            obj_stock.quantite += quantite
        else:
            # Calculer les prix
            prix_vente = prix_special or int(objet.valeur * self.marge)
            prix_achat = int(objet.valeur * 0.7)  # Le marchand achète moins cher
            
            # Copier l'objet pour le stock
            obj_stock = copy.deepcopy(objet)
            obj_stock.quantite = quantite
            
            # Ajouter au stock
            self.inventaire[objet.id] = (obj_stock, prix_vente, prix_achat)
            
    def calculer_prix_achat(self, objet: Objet, relation_client: int) -> int:
        """Calcule le prix auquel le marchand achète un objet au joueur"""
        if objet.id in self.inventaire:
            _, _, prix_base = self.inventaire[objet.id]
        else:
            prix_base = int(objet.valeur * 0.7)
            
        # Ajustement selon la relation
        modificateur = 1.0 + ((relation_client - 50) / 200)  # ±25% basé sur la relation
        return max(1, int(prix_base * modificateur))
        
    def calculer_prix_vente(self, objet_id: str, relation_client: int) -> int:
        """Calcule le prix de vente d'un objet au joueur"""
        if objet_id not in self.inventaire:
            return 0
            
        _, prix_base, _ = self.inventaire[objet_id]
        
        # Ajustement selon la relation
        modificateur = 1.0 - ((relation_client - 50) / 200)  # ±25% basé sur la relation
        return max(1, int(prix_base * modificateur))
        
    def vendre_au_joueur(self, objet_id: str, quantite: int, inventaire_joueur: Inventaire, 
                         relation_client: int) -> Dict[str, Any]:
        """Vend un objet du marchand au joueur"""
        if objet_id not in self.inventaire:
            return {"succes": False, "message": "Cet objet n'est pas disponible."}
            
        obj_stock, _, _ = self.inventaire[objet_id]
        if obj_stock.quantite < quantite:
            return {"succes": False, "message": f"Stock insuffisant. Disponible: {obj_stock.quantite}"}
            
        prix_total = self.calculer_prix_vente(objet_id, relation_client) * quantite
        if inventaire_joueur.or_possede < prix_total:
            return {"succes": False, "message": f"Fonds insuffisants. Nécessaire: {prix_total} or."}
            
        # Créer l'objet pour le joueur
        objet_a_vendre = copy.deepcopy(obj_stock)
        objet_a_vendre.quantite = quantite
        
        # Transférer l'objet
        if inventaire_joueur.ajouter_objet(objet_a_vendre):
            # Déduire du stock du marchand
            obj_stock.quantite -= quantite
            if obj_stock.quantite <= 0:
                del self.inventaire[objet_id]
                
            # Transfert d'argent
            inventaire_joueur.or_possede -= prix_total
            self.capital += prix_total
            
            return {
                "succes": True,
                "message": f"Achat de {quantite} {objet_a_vendre.nom} pour {prix_total} or.",
                "objet": objet_a_vendre,
                "cout": prix_total
            }
        else:
            return {"succes": False, "message": "Inventaire plein ou poids maximum dépassé."}

# Système de crafting et d'amélioration
@dataclass
class RecetteCraft:
    id: str
    nom: str
    resultat_id: str  # ID de l'objet créé
    ingredients: Dict[str, int]  # ID objet -> quantité
    outils_requis: List[str] = field(default_factory=list)
    competence_requise: Optional[str] = None
    niveau_competence: int = 0
    temps_fabrication: int = 60  # en secondes
    chance_succes: int = 100  # pourcentage
    
    def verifier_ingredients(self, inventaire: Inventaire) -> bool:
        """Vérifie si l'inventaire contient tous les ingrédients nécessaires"""
        for id_ingredient, quantite in self.ingredients.items():
            if id_ingredient not in inventaire.objets:
                return False
            if inventaire.objets[id_ingredient].quantite < quantite:
                return False
        return True
        
    def verifier_outils(self, inventaire: Inventaire) -> bool:
        """Vérifie si l'inventaire contient tous les outils nécessaires"""
        for id_outil in self.outils_requis:
            if id_outil not in inventaire.objets:
                return False
        return True

class GestionnaireCraft:
    def __init__(self):
        self.recettes = {}  # id -> RecetteCraft
        self.objets_creables = {}  # id objet -> id recette
        
    def ajouter_recette(self, recette: RecetteCraft):
        """Ajoute une recette au système de crafting"""
        self.recettes[recette.id] = recette
        self.objets_creables[recette.resultat_id] = recette.id
        
    def fabriquer(self, recette_id: str, personnage: 'Personnage', 
                inventaire: Inventaire) -> Dict[str, Any]:
        """Tente de fabriquer un objet selon une recette"""
        if recette_id not in self.recettes:
            return {"succes": False, "message": "Recette inconnue."}
            
        recette = self.recettes[recette_id]
        
        # Vérifier les prérequis
        if not recette.verifier_ingredients(inventaire):
            return {"succes": False, "message": "Ingrédients insuffisants."}
            
        if not recette.verifier_outils(inventaire):
            return {"succes": False, "message": "Outils requis manquants."}
            
        # Vérifier la compétence
        if recette.competence_requise:
            competence = personnage.competences.get(recette.competence_requise)
            if not competence or competence.niveau < recette.niveau_competence:
                return {
                    "succes": False, 
                    "message": f"Niveau de compétence en {recette.competence_requise} insuffisant."
                }
        
        # Consommer les ingrédients
        for id_ingredient, quantite in recette.ingredients.items():
            inventaire.retirer_objet(id_ingredient, quantite)
            
        # Déterminer le succès
        chance_ajustee = recette.chance_succes
        if recette.competence_requise:
            competence = personnage.competences.get(recette.competence_requise)
            if competence:
                # Bonus de compétence
                bonus = (competence.niveau - recette.niveau_competence) * 5
                chance_ajustee = min(100, chance_ajustee + bonus)
                
        succes = random.randint(1, 100) <= chance_ajustee
        
        if succes:
            # Créer l'objet résultat
            # (Nécessiterait un système pour instancier les objets à partir des IDs)
            return {
                "succes": True,
                "message": f"Fabrication de {recette.nom} réussie!",
                "objet_id": recette.resultat_id
            }
        else:
            return {"succes": False, "message": "La fabrication a échoué..."}

# =================================================================
# === BLOC 7/12 : SYSTÈME DE DIALOGUE ET INTERACTIONS SOCIALES ===
# =================================================================

# Enums pour le système de dialogue
class TypeDialogue(Enum):
    NORMAL = auto()
    IMPORTANT = auto()
    QUETE = auto()
    COMMERCE = auto()
    COMBAT = auto()
    SPECIAL = auto()

class TonDialogue(Enum):
    NEUTRE = auto()
    AMICAL = auto()
    HOSTILE = auto()
    FORMEL = auto()
    INTIME = auto()
    HUMORISTIQUE = auto()
    SARCASTIQUE = auto()
    TRISTE = auto()

# Structure pour les options de dialogue
@dataclass
class OptionDialogue:
    id: str
    texte: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    consequences: Dict[str, Any] = field(default_factory=dict)
    reponse_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ton: TonDialogue = TonDialogue.NEUTRE
    
    def verifier_conditions(self, personnage: 'Personnage', contexte: Dict[str, Any]) -> bool:
        """Vérifie si cette option est disponible selon les conditions"""
        for condition, valeur in self.conditions.items():
            # Vérification d'attributs du personnage
            if condition.startswith("attribut.") and hasattr(personnage, condition[9:]):
                attribut = getattr(personnage, condition[9:])
                if isinstance(attribut, (int, float)) and attribut < valeur:
                    return False
                    
            # Vérification de compétences
            elif condition.startswith("competence."):
                nom_competence = condition[11:]
                if nom_competence not in personnage.competences:
                    return False
                if personnage.competences[nom_competence].niveau < valeur:
                    return False
                    
            # Vérification de relation avec un PNJ
            elif condition.startswith("relation."):
                id_pnj = condition[9:]
                if id_pnj not in personnage.relations:
                    return False
                if personnage.relations[id_pnj].affinite < valeur:
                    return False
                    
            # Vérification de quêtes
            elif condition.startswith("quete."):
                id_quete = condition[6:]
                if condition.endswith(".active"):
                    if id_quete not in personnage.quetes_actives:
                        return False
                elif condition.endswith(".complete"):
                    if id_quete not in personnage.quetes_terminees:
                        return False
                        
            # Vérification d'objets
            elif condition.startswith("objet."):
                id_objet = condition[6:]
                # Cette vérification nécessiterait l'accès à l'inventaire
                
            # Vérification de contexte
            elif condition in contexte:
                if contexte[condition] != valeur:
                    return False
                    
        return True

@dataclass
class ReponseDialogue:
    id: str
    texte: str
    type: TypeDialogue = TypeDialogue.NORMAL
    options_suivantes: List[str] = field(default_factory=list)  # IDs des options
    declencheurs: Dict[str, Any] = field(default_factory=dict)  # Actions déclenchées
    emotions: Dict[str, int] = field(default_factory=dict)  # Emotions exprimées (nom -> intensité)

# Structure des dialogues complets
@dataclass
class Dialogue:
    id: str
    pnj_id: str
    introduction: str
    options: Dict[str, OptionDialogue] = field(default_factory=dict)
    reponses: Dict[str, ReponseDialogue] = field(default_factory=dict)
    conditions_disponibilite: Dict[str, Any] = field(default_factory=dict)
    type: TypeDialogue = TypeDialogue.NORMAL
    priorite: int = 0  # Plus élevé = plus prioritaire
    
    # Nouvelles propriétés pour dialogues dynamiques
    variations_contextuelles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    mots_cles: List[str] = field(default_factory=list)

# Système de gestion des dialogues
class GestionnaireDialogue:
    def __init__(self):
        self.dialogues = {}  # id -> Dialogue
        self.dialogues_par_pnj = defaultdict(list)  # pnj_id -> [dialogue_ids]
        self.historique_conversations = defaultdict(list)  # (joueur_id, pnj_id) -> historique
        self.dialogue_actuel = None
        self.option_actuelle = None
        self.reponse_actuelle = None
        self.contexte_conversation = {}
        
    def ajouter_dialogue(self, dialogue: Dialogue):
        """Ajoute un dialogue au système"""
        self.dialogues[dialogue.id] = dialogue
        self.dialogues_par_pnj[dialogue.pnj_id].append(dialogue.id)
        
    def dialogues_disponibles(self, pnj_id: str, joueur: 'Personnage') -> List[Dialogue]:
        """Retourne les dialogues disponibles pour un PNJ donné"""
        disponibles = []
        
        if pnj_id not in self.dialogues_par_pnj:
            return disponibles
            
        for dialogue_id in self.dialogues_par_pnj[pnj_id]:
            dialogue = self.dialogues.get(dialogue_id)
            if dialogue:
                # Vérifier les conditions de disponibilité
                conditions_remplies = True
                for condition, valeur in dialogue.conditions_disponibilite.items():
                    # Logique de vérification similaire à celle des options
                    # À implémenter selon le système
                    pass
                    
                if conditions_remplies:
                    disponibles.append(dialogue)
                    
        # Trier par priorité
        disponibles.sort(key=lambda d: d.priorite, reverse=True)
        return disponibles
    
    def commencer_conversation(self, dialogue_id: str, joueur: 'Personnage', pnj: 'Personnage') -> Dict[str, Any]:
        """Commence une conversation en utilisant un dialogue spécifique"""
        if dialogue_id not in self.dialogues:
            return {"succes": False, "message": "Dialogue non trouvé."}
            
        dialogue = self.dialogues[dialogue_id]
        self.dialogue_actuel = dialogue
        self.option_actuelle = None
        self.reponse_actuelle = None
        
        # Initialiser le contexte de conversation
        self.contexte_conversation = {
            "joueur": joueur,
            "pnj": pnj,
            "relation": joueur.relations.get(pnj.id, None),
            "premiere_rencontre": pnj.id not in joueur.relations,
            "heure_journee": "matin",  # À remplacer par l'heure réelle du jeu
            "lieu": joueur.position_actuelle.get("lieu", "inconnu"),
        }
        
        # Adapter l'introduction selon le contexte
        introduction = self._adapter_texte(dialogue.introduction, joueur, pnj)
        
        # Récupérer les options disponibles
        options_dispo = self._options_disponibles(dialogue, joueur)
        
        # Enregistrer le début de la conversation dans l'historique
        cle_historique = (joueur.id, pnj.id)
        self.historique_conversations[cle_historique].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "debut",
            "dialogue_id": dialogue_id,
            "texte": introduction
        })
        
        return {
            "succes": True,
            "introduction": introduction,
            "options": [
                {"id": opt.id, "texte": opt.texte} 
                for opt in options_dispo
            ]
        }
        
    def _options_disponibles(self, dialogue: Dialogue, personnage: 'Personnage') -> List[OptionDialogue]:
        """Retourne les options de dialogue disponibles selon le contexte actuel"""
        options = []
        
        for opt_id, option in dialogue.options.items():
            if self.option_actuelle is None or opt_id in self.reponse_actuelle.options_suivantes:
                if option.verifier_conditions(personnage, self.contexte_conversation):
                    options.append(option)
                    
        return options
    
    def _adapter_texte(self, texte: str, joueur: 'Personnage', pnj: 'Personnage') -> str:
        """Adapte le texte en remplaçant les variables et en tenant compte du contexte"""
        # Remplacer les variables
        texte = texte.replace("{JOUEUR}", joueur.nom)
        texte = texte.replace("{PNJ}", pnj.nom)
        
        # Remplacer d'autres variables selon le contexte
        for cle, valeur in self.contexte_conversation.items():
            if isinstance(valeur, str):
                texte = texte.replace(f"{{{cle.upper()}}}", valeur)
        
        # Appliquer les variations contextuelles
        if self.dialogue_actuel:
            for contexte, variations in self.dialogue_actuel.variations_contextuelles.items():
                valeur_contexte = self.contexte_conversation.get(contexte)
                if valeur_contexte and valeur_contexte in variations:
                    pattern = f"\\{{{contexte.upper()}:([^}}]*)\\}}"
                    replacement = variations[valeur_contexte]
                    texte = re.sub(pattern, replacement, texte)
        
        return texte
        
    def choisir_option(self, option_id: str, joueur: 'Personnage', pnj: 'Personnage') -> Dict[str, Any]:
        """Traite le choix d'une option de dialogue par le joueur"""
        if not self.dialogue_actuel or option_id not in self.dialogue_actuel.options:
            return {"succes": False, "message": "Option non disponible."}
            
        option = self.dialogue_actuel.options[option_id]
        self.option_actuelle = option
        
        # Appliquer les conséquences de l'option
        for consequence, valeur in option.consequences.items():
            # Modification de relation
            if consequence.startswith("relation."):
                id_cible = consequence.split('.')[1]
                aspect = consequence.split('.')[2] if len(consequence.split('.')) > 2 else "affinite"
                
                if id_cible == pnj.id and id_cible in joueur.relations:
                    joueur.relations[id_cible].modifier_relation(aspect, valeur, f"Option de dialogue: {option.texte}")
            
            # Autres types de conséquences possibles...
        
        # Récupérer la réponse associée
        if not option.reponse_id or option.reponse_id not in self.dialogue_actuel.reponses:
            return {"succes": False, "message": "Pas de réponse associée."}
            
        reponse = self.dialogue_actuel.reponses[option.reponse_id]
        self.reponse_actuelle = reponse
        
        # Adapter le texte de réponse
        texte_reponse = self._adapter_texte(reponse.texte, joueur, pnj)
        
        # Déclencher les actions associées à cette réponse
        for declencheur, params in reponse.declencheurs.items():
            if declencheur == "quete":
                # Logique pour démarrer une quête
                pass
            elif declencheur == "commerce":
                # Logique pour ouvrir l'interface de commerce
                pass
            # Autres déclencheurs...
        
        # Récupérer les nouvelles options disponibles
        options_dispo = self._options_disponibles(self.dialogue_actuel, joueur)
        
        # Enregistrer dans l'historique
        cle_historique = (joueur.id, pnj.id)
        self.historique_conversations[cle_historique].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "option",
            "texte": option.texte,
            "option_id": option.id
        })
        self.historique_conversations[cle_historique].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "reponse",
            "texte": texte_reponse,
            "reponse_id": reponse.id,
            "emotions": reponse.emotions
        })
        
        # Mettre à jour le contexte avec les émotions exprimées
        for emotion, intensite in reponse.emotions.items():
            self.contexte_conversation[f"emotion_{emotion}"] = intensite
        
        return {
            "succes": True,
            "reponse": texte_reponse,
            "emotions": reponse.emotions,
            "options": [
                {"id": opt.id, "texte": opt.texte} 
                for opt in options_dispo
            ],
            "fin_dialogue": len(options_dispo) == 0
        }

# Système de négociation et persuasion
class SystemePersuasion:
    def __init__(self):
        self.difficulte_base = {
            "facile": 20,
            "moyenne": 40,
            "difficile": 60,
            "tres_difficile": 80
        }
    
    def tenter_persuasion(self, personnage: 'Personnage', cible: 'Personnage', 
                        type_persuasion: str, contexte: Dict[str, Any]) -> Dict[str, Any]:
        """Tente de persuader un PNJ"""
        # Base de difficulté selon le contexte
        difficulte = self.difficulte_base.get(contexte.get("difficulte", "moyenne"), 40)
        
        # Facteurs du personnage
        niveau_charisme = personnage.charisme
        niveau_intelligence = personnage.intelligence
        
        # Bonus de compétences spécifiques
        bonus_competence = 0
        if "persuasion" in personnage.competences:
            bonus_competence += personnage.competences["persuasion"].niveau * 2
        if "negociation" in personnage.competences and type_persuasion == "negociation":
            bonus_competence += personnage.competences["negociation"].niveau * 3
            
        # Facteurs de relation
        bonus_relation = 0
        if cible.id in personnage.relations:
            relation = personnage.relations[cible.id]
            bonus_relation = relation.affinite // 10
            
        # Facteurs de contexte supplémentaires
        bonus_contexte = 0
        if contexte.get("pot_de_vin") and isinstance(contexte["pot_de_vin"], int):
            bonus_contexte += min(30, contexte["pot_de_vin"] // 10)
        if contexte.get("menace") and cible.traits.primaires.get("couardise", 0) > 5:
            bonus_contexte += 15
            
        # Calcul de la chance de réussite
        chance_base = niveau_charisme + niveau_intelligence//2 + bonus_competence + bonus_relation + bonus_contexte
        chance_finale = min(95, max(5, chance_base - difficulte))
        
        # Déterminer le résultat
        reussite = random.randint(1, 100) <= chance_finale
        
        # Déterminer l'ampleur du succès ou de l'échec
        marge = chance_finale - (0 if reussite else 100)
        ampleur = "critique" if abs(marge) > 30 else "normale"
        
        # Résultats
        if reussite:
            # Augmenter légèrement la relation en cas de succès
            if cible.id in personnage.relations:
                personnage.relations[cible.id].modifier_relation("respect", 2, "Persuasion réussie")
                
            # Augmenter l'expérience de compétence
            if "persuasion" in personnage.competences:
                personnage.competences["persuasion"].experience += 5
                
            message = "Votre argument convainc votre interlocuteur."
            if ampleur == "critique":
                message = "Votre argument est particulièrement persuasif!"
        else:
            # Diminuer légèrement la relation en cas d'échec
            if cible.id in personnage.relations:
                personnage.relations[cible.id].modifier_relation("respect", -1, "Persuasion échouée")
                
            # Un peu d'expérience quand même
            if "persuasion" in personnage.competences:
                personnage.competences["persuasion"].experience += 2
                
            message = "Votre argument ne semble pas convaincre."
            if ampleur == "critique":
                message = "Votre argument irrite profondément votre interlocuteur!"
        
        return {
            "succes": reussite,
            "message": message,
            "ampleur": ampleur,
            "chance_calculee": chance_finale
        }

# Système d'émotions et réactions PNJ
@dataclass
class EtatEmotionnel:
    emotions: Dict[str, int] = field(default_factory=lambda: {
        "joie": 0, "colère": 0, "peur": 0, "tristesse": 0, 
        "dégoût": 0, "surprise": 0, "confiance": 0, "anticipation": 0
    })
    humeur_base: str = "neutre"
    seuil_reaction: int = 70  # Intensité à partir de laquelle une réaction est déclenchée
    duree_persistance: int = 5  # Tours/interactions avant dissipation
    
    def emotion_dominante(self) -> Tuple[str, int]:
        """Retourne l'émotion la plus forte et son intensité"""
        return max(self.emotions.items(), key=lambda x: x[1], default=("neutre", 0))
        
    def ajouter_emotion(self, emotion: str, intensite: int):
        """Ajoute une émotion ou augmente son intensité"""
        if emotion in self.emotions:
            self.emotions[emotion] = min(100, self.emotions[emotion] + intensite)
            
    def attenuer_emotions(self):
        """Atténue progressivement toutes les émotions"""
        for emotion in self.emotions:
            self.emotions[emotion] = max(0, self.emotions[emotion] - 10)
            
    def generer_reaction(self) -> Optional[Dict[str, Any]]:
        """Génère une réaction basée sur l'état émotionnel actuel"""
        emotion, intensite = self.emotion_dominante()
        
        if intensite < self.seuil_reaction:
            return None
            
        reactions = {
            "colère": ["froncer les sourcils", "serrer les poings", "hausser le ton"],
            "joie": ["sourire", "rire légèrement", "avoir les yeux qui pétillent"],
            "peur": ["reculer", "regarder nerveusement autour", "transpirer"],
            "tristesse": ["baisser les yeux", "soupirer", "parler doucement"],
            # etc.
        }
        
        if emotion in reactions:
            reaction = random.choice(reactions[emotion])
            return {"emotion": emotion, "intensite": intensite, "reaction": reaction}
        
        return None

# Système de réputation sociale
@dataclass
class ReputationSociale:
    regions: Dict[str, int] = field(default_factory=dict)  # région -> valeur
    factions: Dict[str, int] = field(default_factory=dict)  # faction -> valeur
    classes_sociales: Dict[str, int] = field(default_factory=dict)  # classe -> valeur
    titres: List[str] = field(default_factory=list)
    
    # Événements notables qui ont affecté la réputation
    evenements_notables: List[Dict[str, Any]] = field(default_factory=list)
    
    def modifier_reputation(self, categorie: str, cible: str, valeur: int, raison: str = ""):
        """Modifie la réputation auprès d'une cible spécifique"""
        if categorie == "region":
            self.regions[cible] = max(-100, min(100, self.regions.get(cible, 0) + valeur))
        elif categorie == "faction":
            self.factions[cible] = max(-100, min(100, self.factions.get(cible, 0) + valeur))
        elif categorie == "classe":
            self.classes_sociales[cible] = max(-100, min(100, self.classes_sociales.get(cible, 0) + valeur))
            
        # Enregistrer l'événement si significatif
        if abs(valeur) >= 5:
            self.evenements_notables.append({
                "date": datetime.datetime.now().isoformat(),
                "categorie": categorie,
                "cible": cible,
                "modification": valeur,
                "raison": raison
            })
    
    def obtenir_reputation(self, categorie: str, cible: str) -> int:
        """Récupère la valeur de réputation pour une cible spécifique"""
        if categorie == "region":
            return self.regions.get(cible, 0)
        elif categorie == "faction":
            return self.factions.get(cible, 0)
        elif categorie == "classe":
            return self.classes_sociales.get(cible, 0)
        return 0
        
    def description_reputation(self, categorie: str, cible: str) -> str:
        """Retourne une description qualitative de la réputation"""
        valeur = self.obtenir_reputation(categorie, cible)
        
        if valeur > 80:
            return "Légendaire"
        elif valeur > 60:
            return "Admiré"
        elif valeur > 40:
            return "Respecté"
        elif valeur > 20:
            return "Apprécié"
        elif valeur > -20:
            return "Neutre"
        elif valeur > -40:
            return "Méfiance"
        elif valeur > -60:
            return "Méprisé"
        elif valeur > -80:
            return "Haï"
        else:
            return "Ennemi juré"

# =================================================================
# === BLOC 8/12 : SYSTÈME D'IA ET NARRATION DYNAMIQUE ===
# =================================================================

# Configuration de l'interface avec l'IA
class ConfigurationIA:
    def __init__(self, modele: str = DEFAULT_MODEL):
        self.modele = modele
        self.temperature = 0.7
        self.max_tokens = 2000
        self.presence_penalty = 0.3
        self.frequency_penalty = 0.5
        self.historique_max = 20  # Nombre max d'échanges dans l'historique
        self.instructions_systeme = ""
        self.contexte_monde = ""
        self.style_narratif = "descriptif"  # descriptif, dramatique, poétique, etc.
        
    def definir_style_narratif(self, style: str, intensite: float = 1.0):
        """Change le style narratif de l'IA"""
        styles_disponibles = {
            "descriptif": "Fournis des descriptions détaillées, objectives et immersives.",
            "dramatique": "Utilise un ton dramatique, intense, avec des enjeux élevés.",
            "poetique": "Emploie des métaphores, du lyrisme et un langage évocateur.",
            "humoristique": "Intègre des touches d'humour, d'ironie et de légèreté.",
            "mystérieux": "Favorise l'ambiguïté, les questions sans réponses et l'atmosphère intrigante.",
            "action": "Dynamique, phrases courtes, verbes forts, rythme rapide.",
            "philosophique": "Inclus des réflexions, questions existentielles et perspectives profondes."
        }
        
        if style in styles_disponibles:
            self.style_narratif = style
            instruction = styles_disponibles[style]
            
            # Ajuster selon l'intensité
            if intensite > 1.5:
                instruction += " Exagère fortement ce style."
            elif intensite > 1.0:
                instruction += " Accentue ce style."
            elif intensite < 0.8:
                instruction += " Applique ce style avec subtilité."
                
            # Intégrer dans les instructions système
            self._mettre_a_jour_instructions_style(style, instruction)
    
    def _mettre_a_jour_instructions_style(self, style: str, instruction: str):
        """Met à jour la partie style dans les instructions système"""
        # Remplacer les instructions de style existantes ou en ajouter
        style_pattern = r"STYLE NARRATIF:.*?(?=\n\n|\Z)"
        nouvelle_instruction = f"STYLE NARRATIF: {style}. {instruction}"
        
        if re.search(style_pattern, self.instructions_systeme, re.DOTALL):
            self.instructions_systeme = re.sub(
                style_pattern, 
                nouvelle_instruction, 
                self.instructions_systeme, 
                flags=re.DOTALL
            )
        else:
            self.instructions_systeme += f"\n\n{nouvelle_instruction}"

# Gestionnaire principal d'interaction avec l'IA
class GestionnaireIA:
    def __init__(self, configuration: ConfigurationIA = None):
        self.configuration = configuration or ConfigurationIA()
        self.historique_messages = []
        self.client = client  # Utilise le client OpenAI déjà configuré
        self.instruction_specialisees = {}  # Contextes spéciaux -> instructions
    
    def initialiser_contexte(self, personnage: 'Personnage', monde: Dict[str, Any]):
        """Initialise le contexte global pour l'IA"""
        # Construction du contexte du monde
        contexte_monde = f"""
        UNIVERS: Mushoku Tensei - Un monde de magie, d'aventure et de réincarnation.
        
        DATE ACTUELLE: {monde.get('date_actuelle', 'Inconnue')}
        LIEU ACTUEL: {personnage.position_actuelle.get('lieu', 'Inconnu')} dans {personnage.position_actuelle.get('region', 'région inconnue')}
        
        CONTEXTE MONDIAL:
        - Système magique basé sur les éléments: Eau, Feu, Terre, Vent, etc.
        - Différentes races: Humains, Elfes, Nains, Beastfolk, Supards, Démons, etc.
        - Tensions politiques entre le Royaume de Milis, l'Empire d'Asura, et d'autres nations.
        - La magie de téléportation a causé une grande catastrophe par le passé.
        - Les Sept Grandes Puissances Mondiales sont des combattants d'élite avec des titres spéciaux.
        
        PERSONNALITÉS IMPORTANTES:
        - Les dieux-dragons: Orsted, Hitogami, etc.
        - Les dirigeants des nations majeures
        - Les détenteurs des titres des Sept Grandes Puissances Mondiales
        """
        
        # Contexte spécifique du personnage
        traits_principaux = ", ".join([f"{trait}: {valeur}" for trait, valeur in list(personnage.traits.primaires.items())[:3]])
        competences_principales = ", ".join([comp.nom for comp in sorted(personnage.competences.values(), key=lambda c: c.niveau, reverse=True)[:3]])
        
        contexte_personnage = f"""
        PERSONNAGE PRINCIPAL:
        Nom: {personnage.nom}
        Race: {personnage.race}
        Âge: {personnage.age}
        Traits principaux: {traits_principaux}
        Compétences principales: {competences_principales}
        Objectifs actuels: {', '.join(obj.get('description', 'Inconnu') for obj in personnage.objectifs_personnels[:2])}
        """
        
        # Assembler le contexte complet
        self.configuration.contexte_monde = contexte_monde + "\n" + contexte_personnage
        
        # Réinitialiser l'historique
        self.historique_messages = []
        
        # Définir les instructions système de base
        self.configuration.instructions_systeme = f"""
        Tu es un narrateur pour une aventure roleplay dans l'univers de Mushoku Tensei. Ton but est de créer une expérience immersive, captivante et cohérente avec le lore établi.

        DIRECTIVES PRINCIPALES:
        1. Respecte fidèlement le lore de Mushoku Tensei et les éléments déjà établis.
        2. Adapte les descriptions aux actions et choix du joueur.
        3. Sois cohérent avec les personnages canoniques, leurs motivations et personnalités.
        4. Équilibre descriptions, dialogues et narration pour maintenir le rythme.
        5. Inclus des détails sensoriels pour renforcer l'immersion.
        6. Propose des choix et conséquences significatifs qui respectent le libre arbitre.
        7. Adapte le ton aux situations (danger, romance, mystère, etc.).
        
        STYLE NARRATIF: {self.configuration.style_narratif}. Fournis des descriptions détaillées et immersives.
        
        N'introduis pas de joueur nommé Rudeus Greyrat, car le joueur incarne son propre personnage original dans ce monde.
        """
    
    def ajouter_contexte_specialise(self, type_contexte: str, contenu: str):
        """Ajoute un contexte spécialisé pour certaines situations"""
        self.instruction_specialisees[type_contexte] = contenu
    
    def generer_narration(self, prompt: str, contexte_specifique: Dict[str, Any] = None) -> str:
        """Génère une narration basée sur un prompt et le contexte actuel"""
        # Préparer les messages pour l'IA
        messages = self._preparer_messages(prompt, contexte_specifique)
        
        # Appeler l'IA
        try:
            response = self.client.chat.completions.create(
                model=self.configuration.modele,
                messages=messages,
                max_tokens=self.configuration.max_tokens,
                temperature=self.configuration.temperature,
                presence_penalty=self.configuration.presence_penalty,
                frequency_penalty=self.configuration.frequency_penalty
            )
            
            narration = response.choices[0].message.content
            
            # Ajouter à l'historique
            self.historique_messages.append({"role": "user", "content": prompt})
            self.historique_messages.append({"role": "assistant", "content": narration})
            
            # Tronquer l'historique si nécessaire
            if len(self.historique_messages) > self.configuration.historique_max * 2:
                self.historique_messages = self.historique_messages[-self.configuration.historique_max * 2:]
                
            return narration
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de narration: {str(e)}")
            return f"[Erreur de génération: {str(e)}]"
    
    def _preparer_messages(self, prompt: str, contexte_specifique: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Prépare la liste de messages pour l'appel à l'API"""
        messages = []
        
        # Message système avec instructions de base
        instructions = self.configuration.instructions_systeme
        
        # Ajouter le contexte du monde
        if self.configuration.contexte_monde:
            instructions += f"\n\nCONTEXTE ACTUEL:\n{self.configuration.contexte_monde}"
        
        # Ajouter instructions spécialisées selon le contexte
        if contexte_specifique:
            for type_ctx, valeur in contexte_specifique.items():
                if type_ctx in self.instruction_specialisees:
                    instructions += f"\n\n{self.instruction_specialisees[type_ctx]}"
                    
            # Contextes spécifiques courants
            if "combat" in contexte_specifique and contexte_specifique["combat"]:
                instructions += "\n\nSITUATION DE COMBAT: Décris l'action avec dynamisme, précision et tension. Mets l'accent sur les techniques, les impacts et les réactions des combattants."
                
            if "lieu_important" in contexte_specifique and contexte_specifique["lieu_important"]:
                instructions += f"\n\nDESCRIPTION DE LIEU IMPORTANT: Accorde une attention particulière aux détails architecturaux, à l'histoire et à l'ambiance du lieu: {contexte_specifique['lieu_important']}"
        
        messages.append({"role": "system", "content": instructions})
        
        # Ajouter l'historique récent
        messages.extend(self.historique_messages[-self.configuration.historique_max * 2:])
        
        # Ajouter le prompt actuel
        messages.append({"role": "user", "content": prompt})
        
        return messages

    def generer_reve(self, personnage: 'Personnage', importance: int = 5, 
                   type_reve: str = "standard") -> Dict[str, Any]:
        """Génère un rêve pour le personnage basé sur ses expériences récentes"""
        # Récupérer contexte pour le rêve
        evenements_recents = personnage.historique_evenements[-5:] if personnage.historique_evenements else []
        evenements_texte = "\n".join([f"- {ev.get('type', 'événement')}: {ev.get('details', 'aucun détail')}" 
                                    for ev in evenements_recents])
        
        emotions_dominantes = sorted(
            personnage.traits.memoire_emotionnelle.items(), 
            key=lambda x: x[1].get("intensite", 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )[:3]
        emotions_texte = ", ".join([f"{emotion}" for emotion, _ in emotions_dominantes]) if emotions_dominantes else "neutres"
        
        # Construire le prompt pour l'IA
        prompt = f"""
        Génère un rêve pour le personnage sur la base de ses expériences récentes et de son état émotionnel.
        
        IMPORTANCE DU RÊVE: {importance}/10 (plus c'est élevé, plus le rêve est significatif)
        TYPE DE RÊVE: {type_reve} (standard, prophétique, souvenir, cauchemar, etc.)
        
        ÉVÉNEMENTS RÉCENTS:
        {evenements_texte}
        
        ÉTAT ÉMOTIONNEL:
        Émotions dominantes: {emotions_texte}
        
        FORMAT DU RÊVE:
        - Titre: [Un titre évocateur pour ce rêve]
        - Narration: [Description immersive de l'expérience onirique]
        - Symbolisme: [2-3 éléments symboliques et leur possible signification]
        """
        
        contexte = {
            "type_narration": "reve",
            "importance": importance,
            "type_specifique": type_reve
        }
        
        # Générer le contenu du rêve
        contenu_brut = self.generer_narration(prompt, contexte)
        
        # Extraire les éléments structurés
        titre_match = re.search(r"Titre:\s*(.+?)(?:\n|$)", contenu_brut)
        titre = titre_match.group(1).strip() if titre_match else "Rêve étrange"
        
        narration_match = re.search(r"Narration:\s*(.+?)(?=\n- Symbolisme|\n$)", contenu_brut, re.DOTALL)
        narration = narration_match.group(1).strip() if narration_match else contenu_brut
        
        symbolisme_match = re.search(r"Symbolisme:\s*(.+?)(?:\n|$)", contenu_brut, re.DOTALL)
        symbolisme_texte = symbolisme_match.group(1).strip() if symbolisme_match else ""
        
        # Créer un dictionnaire de symbolisme
        symbolisme = {}
        for ligne in symbolisme_texte.split("\n"):
            if ":" in ligne:
                symbole, signification = ligne.split(":", 1)
                symbolisme[symbole.strip()] = signification.strip()
        
        # Déterminer si le rêve est prophétique
        est_prophetique = type_reve == "prophétique" or "prophét" in contenu_brut.lower()
        
        # Créer l'objet rêve
        reve = {
            "titre": titre,
            "contenu": narration,
            "symbolisme": symbolisme,
            "est_prophetique": est_prophetique,
            "intensite": importance
        }
        
        return reve

# Système de narration dynamique
class NarrateurDynamique:
    def __init__(self, gestionnaire_ia: GestionnaireIA):
        self.ia = gestionnaire_ia
        self.scenes_memorisees = deque(maxlen=10)  # Garde trace des dernières scènes
        self.mots_cles_importants = set()  # Mots-clés récurrents ou significatifs
        self.arcs_narratifs = {}  # Arcs narratifs en cours
        self.tension_narrative = 0  # 0-10, influence le style et le rythme
    
    def generer_description_lieu(self, lieu: Emplacement, personnage: 'Personnage', 
                              premiere_visite: bool = False) -> str:
        """Génère une description adaptée d'un lieu"""
        meteo = lieu.meteo_actuelle
        saison = "inconnue"  # À remplacer par la saison actuelle du monde
        
        caracteristiques_lieu = [
            f"Type: {lieu.type}",
            f"Région: {lieu.region} ({lieu.continent})",
            f"Danger: {lieu.danger}/10",
            f"Population: {'importante' if len(lieu.habitants) > 20 else 'modérée' if len(lieu.habitants) > 5 else 'faible'}"
        ]
        
        services = [service for service, disponible in lieu.services.items() if disponible]
        
        prompt = f"""
        Décris le lieu suivant de façon immersive et adaptée au contexte:
        
        LIEU: {lieu.nom}
        CARACTÉRISTIQUES: {', '.join(caracteristiques_lieu)}
        MÉTÉO ACTUELLE: {meteo.condition.name.lower()}, intensité {meteo.intensite}/10, température {meteo.temperature}°C
        SAISON: {saison}
        
        SERVICES DISPONIBLES: {', '.join(services) if services else 'aucun'}
        
        CONTEXTE POUR LE JOUEUR:
        - {personnage.nom} {f'découvre ce lieu pour la première fois' if premiere_visite else 'est déjà venu ici'}
        - Heure du jour: {personnage.position_actuelle.get('heure', 'jour')}
        
        {lieu.description if lieu.description else ''}
        """
        
        contexte = {
            "type_narration": "description_lieu",
            "premiere_visite": premiere_visite,
            "lieu_important": lieu.nom if lieu.danger >= 7 or "capitale" in lieu.type.lower() else None
        }
        
        return self.ia.generer_narration(prompt, contexte)
    
    def generer_rencontre_pnj(self, pnj: 'Personnage', joueur: 'Personnage', 
                           connaissance_prealable: bool = False) -> str:
        """Génère la narration d'une rencontre avec un PNJ"""
        # Récupérer le niveau de relation s'il existe
        relation = None
        if pnj.id in joueur.relations:
            relation = joueur.relations[pnj.id]
        
        traits_notables = list(pnj.traits.primaires.items())[:3]
        traits_texte = ", ".join([f"{trait}: {valeur}" for trait, valeur in traits_notables])
        
        # Contexte relationnel
        contexte_relation = "première rencontre"
        if relation:
            if relation.affinite > 50:
                contexte_relation = "relation positive"
            elif relation.affinite < -20:
                contexte_relation = "relation négative"
            else:
                contexte_relation = "relation neutre"
                
            if relation.peur > 50:
                contexte_relation += ", intimidation"
                
            if relation.respect > 60:
                contexte_relation += ", respect"
        
        prompt = f"""
        Décris la rencontre entre le personnage joueur et ce PNJ:
        
        PNJ: {pnj.nom}
        TITRE/RÔLE: {pnj.titre if pnj.titre else 'aucun'}
        APPARENCE: {pnj.description if pnj.description else 'À décrire de façon cohérente avec sa race, son âge et son rôle.'}
        TRAITS NOTABLES: {traits_texte}
        
        CONTEXTE:
        - Le PNJ est en train de: [invente une activité appropriée]
        - Relation avec le joueur: {contexte_relation}
        - Cet individu est {'' if pnj.est_canonique else 'non '}canonique dans l'univers de Mushoku Tensei
        
        Focalise-toi sur les détails physiques expressifs, le langage corporel et l'attitude du PNJ envers le joueur.
        Ne fais pas parler directement le PNJ (le dialogue viendra après).
        """
        
        contexte = {
            "type_narration": "rencontre_pnj",
            "canonique": pnj.est_canonique,
            "importance": 8 if pnj.est_canonique else 5
        }
        
        return self.ia.generer_narration(prompt, contexte)
    
    def generer_combat(self, gestionnaire_combat: GestionnaireCombat, tour: int, 
                    derniere_action: Dict[str, Any] = None) -> str:
        """Génère une narration pour un tour de combat"""
        # Extraire les informations du gestionnaire de combat
        participants = []
        for i, p in enumerate(gestionnaire_combat.participants):
            participants.append({
                "nom": p["personnage"].nom,
                "equipe": p["equipe"],
                "sante": f"{p['personnage'].sante_actuelle}/{p['personnage'].sante_max}",
                "est_hors_combat": p["est_hors_combat"]
            })
        
        # Récupérer les logs du tour actuel
        logs_tour = [log for log in gestionnaire_combat.log_combat 
                    if log.get("tour") == tour]
        
        # Construire un résumé des actions
        resume_actions = ""
        for log in logs_tour:
            if log["type"] == "debut_tour":
                continue
                
            if log["type"] == "attaque_normale":
                resume_actions += f"- {log['attaquant']} a attaqué {log['cible']} pour {log['degats']} dégâts.\n"
                if log.get("hors_combat"):
                    resume_actions += f"  → {log['cible']} est mis hors combat!\n"
            
            # Ajouter d'autres types d'actions (sort, technique, etc.)
        
        # Construire le prompt
        prompt = f"""
        Narre le tour {tour} de ce combat de façon dynamique et immersive:
        
        PARTICIPANTS:
        {json.dumps(participants, indent=2)}
        
        RÉSUMÉ DES ACTIONS DE CE TOUR:
        {resume_actions if resume_actions else "Début du combat."}
        
        DERNIÈRE ACTION IMPORTANTE:
        {json.dumps(derniere_action, indent=2) if derniere_action else "Aucune"}
        
        Décris l'action avec dynamisme et tension. Utilise un langage visuel et cinématique. 
        Inclus les réactions émotionnelles, la douleur, l'adrénaline et l'environnement du combat.
        """
        
        contexte = {
            "type_narration": "combat",
            "combat": True,
            "tension": min(10, tour + 3)  # La tension augmente avec les tours
        }
        
        narration = self.ia.generer_narration(prompt, contexte)
        
        # Mémoriser pour continuité narrative
        self.scenes_memorisees.append({
            "type": "combat",
            "tour": tour,
            "narration": narration[:100] + "..." if len(narration) > 100 else narration
        })
        
        return narration
        
    def generer_revelation_quete(self, quete: Quete, declencheur: str = "découverte") -> str:
        """Génère une narration pour la révélation d'une nouvelle quête"""
        objectifs_texte = "\n".join([f"- {obj.description}" for obj in quete.objectifs])
        
        prompt = f"""
        Décris la révélation de cette nouvelle quête de façon captivante:
        
        QUÊTE: {quete.titre}
        TYPE: {quete.type.name}
        DIFFICULTÉ: {quete.difficulte.name}
        
        DESCRIPTION:
        {quete.description}
        
        OBJECTIFS:
        {objectifs_texte}
        
        DÉCLENCHEUR:
        Cette quête a été révélée par: {declencheur}
        
        Crée une narration qui suscite l'intérêt et établit clairement les enjeux de la quête.
        Si la quête vient d'un PNJ, n'inclus pas directement le dialogue (il sera géré séparément).
        """
        
        contexte = {
            "type_narration": "quete",
            "importance": {
                TypeQuete.PRINCIPALE: 9,
                TypeQuete.SECONDAIRE: 6,
                TypeQuete.FACTION: 7,
                TypeQuete.PERSONNELLE: 8,
                TypeQuete.QUOTIDIENNE: 4,
                TypeQuete.MONDE: 8
            }.get(quete.type, 5)
        }
        
        return self.ia.generer_narration(prompt, contexte)
        
    def ajuster_tension_narrative(self, modificateur: int):
        """Ajuste le niveau de tension narrative"""
        self.tension_narrative = max(0, min(10, self.tension_narrative + modificateur))
        
        # Adapter le style narratif selon la tension
        if self.tension_narrative >= 8:
            self.ia.configuration.definir_style_narratif("dramatique", 1.3)
        elif self.tension_narrative >= 6:
            self.ia.configuration.definir_style_narratif("action", 1.2)
        elif self.tension_narrative <= 2:
            self.ia.configuration.definir_style_narratif("descriptif", 1.0)
        else:
            # Tension moyenne - style par défaut
            self.ia.configuration.definir_style_narratif("descriptif", 1.0)

# =================================================================
# === BLOC 9/12 : SYSTÈME DE PROGRESSION ET COMPÉTENCES ===
# =================================================================

# Constantes pour le système de progression
SEUILS_NIVEAU = [0, 100, 300, 600, 1000, 1500, 2100, 2800, 3600, 4500]  # Expérience requise par niveau
SEUILS_COMPETENCE = [0, 50, 150, 300, 500, 750, 1050, 1400, 1800, 2250, 2750]  # Expérience par niveau de compétence
AFFINITE_CATEGORIE = {  # Multiplicateurs d'expérience par catégorie
    "innee": 1.5,       # Talent naturel
    "compatible": 1.2,  # Bonne compatibilité
    "standard": 1.0,    # Progression normale
    "difficile": 0.8,   # Apprentissage difficile
    "incompatible": 0.6  # Très difficile à apprendre
}

# Système de progression de personnage
@dataclass
class SystemeProgression:
    personnage: 'Personnage'
    multiplicateur_xp: float = 1.0  # Modificateur global d'acquisition d'XP
    sources_experience: Dict[str, int] = field(default_factory=lambda: {
        "combat": 0, "quete": 0, "exploration": 0, "social": 0, "artisanat": 0
    })
    
    def ajouter_experience(self, quantite: int, source: str = "divers"):
        """Ajoute de l'expérience au personnage et vérifie le niveau"""
        if source in self.sources_experience:
            self.sources_experience[source] += quantite
        else:
            self.sources_experience["divers"] = self.sources_experience.get("divers", 0) + quantite
            
        # Appliquer le multiplicateur
        xp_ajustee = int(quantite * self.multiplicateur_xp)
        
        ancien_niveau = self.personnage.niveau
        self.personnage.experience += xp_ajustee
        
        # Vérifier si niveau supérieur atteint
        nouveau_niveau = self._calculer_niveau_actuel()
        if nouveau_niveau > ancien_niveau:
            self._appliquer_montee_niveau(ancien_niveau, nouveau_niveau)
            
        return {
            "xp_gagnee": xp_ajustee,
            "nouveau_niveau": nouveau_niveau > ancien_niveau,
            "niveaux_gagnes": nouveau_niveau - ancien_niveau
        }
    
    def _calculer_niveau_actuel(self) -> int:
        """Détermine le niveau actuel en fonction de l'expérience totale"""
        xp_totale = self.personnage.experience
        
        # Trouver le niveau correspondant à l'expérience actuelle
        for niveau, seuil in enumerate(SEUILS_NIVEAU):
            if xp_totale < seuil:
                return max(1, niveau)
                
        # Si au-delà du dernier seuil défini
        return len(SEUILS_NIVEAU)
    
    def _appliquer_montee_niveau(self, ancien: int, nouveau: int):
        """Applique les bonus de montée de niveau"""
        niveaux_gagnes = nouveau - ancien
        self.personnage.niveau = nouveau
        
        # Points de caractéristiques gagnés (3 par niveau)
        points_caracs = niveaux_gagnes * 3
        
        # Points de santé/mana gagnés (calculés selon les attributs)
        bonus_sante = niveaux_gagnes * (5 + self.personnage.endurance // 10)
        bonus_mana = niveaux_gagnes * (3 + self.personnage.magie // 10)
        
        # Appliquer les bonus
        self.personnage.sante_max += bonus_sante
        self.personnage.sante_actuelle = min(self.personnage.sante_actuelle + bonus_sante, self.personnage.sante_max)
        
        self.personnage.mana_max += bonus_mana
        self.personnage.mana_actuel = min(self.personnage.mana_actuel + bonus_mana, self.personnage.mana_max)
        
        # Enregistrer l'événement
        self.personnage.historique_evenements.append({
            "type": "montee_niveau",
            "date": datetime.datetime.now().isoformat(),
            "ancien_niveau": ancien,
            "nouveau_niveau": nouveau,
            "points_caracs_gagnes": points_caracs,
            "bonus_sante": bonus_sante,
            "bonus_mana": bonus_mana
        })
        
        return {
            "points_caracs": points_caracs,
            "bonus_sante": bonus_sante,
            "bonus_mana": bonus_mana
        }
        
    def ajouter_experience_competence(self, nom_competence: str, quantite: int, activite: str = ""):
        """Ajoute de l'expérience à une compétence spécifique"""
        # Vérifier si la compétence existe
        if nom_competence not in self.personnage.competences:
            return {"succes": False, "message": f"Compétence {nom_competence} non trouvée"}
            
        competence = self.personnage.competences[nom_competence]
        ancien_niveau = competence.niveau
        
        # Appliquer l'affinité
        xp_ajustee = int(quantite * competence.affinite * self.multiplicateur_xp)
        competence.experience += xp_ajustee
        competence.usages += 1
        
        # Vérifier si niveau supérieur atteint
        nouveau_niveau = self._calculer_niveau_competence(competence.experience)
        if nouveau_niveau > ancien_niveau:
            competence.niveau = nouveau_niveau
            competence.derniere_evolution = datetime.datetime.now()
            
            # Enregistrer l'événement
            self.personnage.historique_evenements.append({
                "type": "amelioration_competence",
                "date": datetime.datetime.now().isoformat(),
                "competence": nom_competence,
                "ancien_niveau": ancien_niveau,
                "nouveau_niveau": nouveau_niveau,
                "activite": activite
            })
            
            # Vérifier si la compétence peut évoluer
            self._verifier_evolution_competence(competence)
            
        return {
            "succes": True,
            "xp_gagnee": xp_ajustee,
            "nouveau_niveau": nouveau_niveau > ancien_niveau,
            "niveau_actuel": nouveau_niveau
        }
    
    def _calculer_niveau_competence(self, experience: int) -> int:
        """Détermine le niveau d'une compétence en fonction de son expérience"""
        for niveau, seuil in enumerate(SEUILS_COMPETENCE):
            if experience < seuil:
                return max(1, niveau)
                
        # Si au-delà du dernier seuil défini
        return len(SEUILS_COMPETENCE)
        
    def _verifier_evolution_competence(self, competence: Competence):
        """Vérifie si une compétence peut évoluer vers une forme plus avancée"""
        if isinstance(competence, CompetenceMagique):
            # Vérifier l'évolution magique
            specialisation = self._determiner_specialisation_magique(competence)
            if specialisation and not competence.specialisation:
                competence.specialisation = specialisation
                return {"type": "specialisation", "resultat": specialisation}
                
        # Vérifier les chemins d'évolution basés sur l'usage
        if competence.niveau >= 5:  # Niveau minimum pour évoluer
            # Logique spécifique d'évolution selon le type de compétence
            pass
            
        return None
        
    def _determiner_specialisation_magique(self, competence: CompetenceMagique) -> Optional[str]:
        """Détermine la spécialisation d'une compétence magique selon son usage"""
        if competence.niveau < 5:
            return None
            
        # Identifier le style dominant
        styles = competence.style_usage
        style_dominant = max(styles.items(), key=lambda x: x[1])[0]
        
        # Seuil minimum d'usage pour spécialisation
        if styles[style_dominant] < 20:
            return None
            
        specialisations = {
            "offensif": {
                "Eau": "Lames d'Eau",
                "Feu": "Explosion de Flammes",
                "Terre": "Lances de Pierre",
                "Vent": "Lames de Vent",
                # etc.
            },
            "défensif": {
                "Eau": "Bouclier Aquatique",
                "Feu": "Barrière de Flammes",
                "Terre": "Mur de Pierre",
                "Vent": "Voile Aérien",
                # etc.
            },
            "utilitaire": {
                "Eau": "Manipulation de Fluides",
                "Feu": "Contrôle de Température",
                "Terre": "Façonnage de Terrain",
                "Vent": "Détection des Courants",
                # etc.
            },
            "soutien": {
                "Eau": "Guérison Aqueuse",
                "Feu": "Vitalité Enflammée",
                "Terre": "Endurance Tellurique",
                "Vent": "Agilité des Brises",
                # etc.
            }
        }
        
        return specialisations.get(style_dominant, {}).get(competence.element)

# Système d'apprentissage de compétences
class SystemeApprentissage:
    def __init__(self, personnage: 'Personnage'):
        self.personnage = personnage
        self.competences_apprises = []
        self.competences_disponibles = {}  # id -> prérequis
        self.enseignements_recus = []
        
    def apprentissage_possible(self, competence_id: str) -> Dict[str, Any]:
        """Vérifie si une compétence peut être apprise par le personnage"""
        if competence_id in self.personnage.competences:
            return {"possible": False, "raison": "Compétence déjà connue"}
            
        if competence_id not in self.competences_disponibles:
            return {"possible": False, "raison": "Compétence non disponible à l'apprentissage"}
            
        prerequisites = self.competences_disponibles[competence_id]
        
        # Vérifier les prérequis
        for pre_comp, niveau in prerequisites.get("competences", {}).items():
            if pre_comp not in self.personnage.competences:
                return {"possible": False, "raison": f"Compétence prérequise manquante: {pre_comp}"}
            if self.personnage.competences[pre_comp].niveau < niveau:
                return {"possible": False, "raison": f"Niveau insuffisant en {pre_comp} (requis: {niveau})"}
                
        # Vérifier les attributs minimums
        for attribut, valeur in prerequisites.get("attributs", {}).items():
            if not hasattr(self.personnage, attribut) or getattr(self.personnage, attribut) < valeur:
                return {"possible": False, "raison": f"Attribut insuffisant: {attribut} (requis: {valeur})"}
                
        return {"possible": True}
    
    def apprendre_competence(self, competence_id: str, source: str = "auto") -> Dict[str, Any]:
        """Tente d'apprendre une nouvelle compétence"""
        # Vérifier si l'apprentissage est possible
        verification = self.apprentissage_possible(competence_id)
        if not verification["possible"]:
            return {"succes": False, "raison": verification["raison"]}
            
        # Récupérer les informations de la compétence (à implémenter selon votre système)
        infos_competence = self._recuperer_infos_competence(competence_id)
        if not infos_competence:
            return {"succes": False, "raison": "Informations sur la compétence non disponibles"}
            
        # Créer la compétence
        nouvelle_competence = Competence(
            nom=infos_competence["nom"],
            description=infos_competence["description"],
            niveau=1,
            affinite=self._calculer_affinite(infos_competence)
        )
        
        # Pour les compétences magiques
        if infos_competence.get("est_magique"):
            nouvelle_competence = CompetenceMagique(
                nom=infos_competence["nom"],
                description=infos_competence["description"],
                niveau=1,
                affinite=self._calculer_affinite(infos_competence),
                element=infos_competence.get("element", ""),
                rang=infos_competence.get("rang", "Débutant"),
                cout_mana=infos_competence.get("cout_mana", 5)
            )
            
        # Ajouter la compétence au personnage
        self.personnage.competences[competence_id] = nouvelle_competence
        
        # Enregistrer l'apprentissage
        self.competences_apprises.append({
            "competence_id": competence_id,
            "date": datetime.datetime.now().isoformat(),
            "source": source
        })
        
        # Enregistrer dans l'historique du personnage
        self.personnage.historique_evenements.append({
            "type": "apprentissage_competence",
            "competence": infos_competence["nom"],
            "date": datetime.datetime.now().isoformat(),
            "source": source
        })
        
        return {"succes": True, "competence": nouvelle_competence}
    
    def _calculer_affinite(self, infos_competence: Dict[str, Any]) -> float:
        """Calcule l'affinité du personnage pour une compétence spécifique"""
        categorie = infos_competence.get("categorie_affinite", "standard")
        multiplicateur_base = AFFINITE_CATEGORIE.get(categorie, 1.0)
        
        # Bonus d'attributs spécifiques
        bonus = 0.0
        for attribut, influence in infos_competence.get("attributs_influents", {}).items():
            if hasattr(self.personnage, attribut):
                valeur = getattr(self.personnage, attribut)
                # +0.01 par point d'attribut au-dessus de 10, pondéré par l'influence
                bonus += max(0, (valeur - 10) / 100) * influence
                
        # Pénalité pour talents contradictoires
        penalite = 0.0
        for trait, malus in infos_competence.get("traits_contradictoires", {}).items():
            if trait in self.personnage.traits.primaires:
                valeur = self.personnage.traits.primaires[trait]
                penalite += (valeur / 100) * malus
                
        return max(0.5, multiplicateur_base + bonus - penalite)  # Minimum 0.5
        
    def _recuperer_infos_competence(self, competence_id: str) -> Dict[str, Any]:
        """Récupère les informations sur une compétence (à implémenter)"""
        # Cette fonction devrait récupérer les informations depuis une base de données
        # ou un fichier de configuration. Pour l'exemple, on utilise des données factices.
        
        competences_disponibles = {
            "boule_feu": {
                "nom": "Boule de Feu",
                "description": "Lance une boule de feu qui explose au contact",
                "est_magique": True,
                "element": "Feu",
                "rang": "Débutant",
                "cout_mana": 8,
                "categorie_affinite": "standard",
                "attributs_influents": {"magie": 1.0, "intelligence": 0.5}
            },
            "epee_basique": {
                "nom": "Techniques d'Épée Basiques",
                "description": "Techniques fondamentales de combat à l'épée",
                "est_magique": False,
                "categorie_affinite": "standard",
                "attributs_influents": {"force": 0.7, "vitesse": 0.3}
            }
            # Plus de compétences...
        }
        
        return competences_disponibles.get(competence_id, None)
    
    def peut_enseigner(self, competence_id: str, eleve: 'Personnage') -> Dict[str, Any]:
        """Vérifie si le personnage peut enseigner une compétence à un autre"""
        # Vérifier si le personnage connaît la compétence
        if competence_id not in self.personnage.competences:
            return {"possible": False, "raison": "Vous ne connaissez pas cette compétence"}
            
        competence = self.personnage.competences[competence_id]
        
        # Vérifier niveau minimum pour enseigner
        if competence.niveau < 3:
            return {"possible": False, "raison": f"Niveau insuffisant en {competence.nom} pour enseigner (minimum 3)"}
            
        # Vérifier si l'élève possède déjà la compétence
        if competence_id in eleve.competences:
            if eleve.competences[competence_id].niveau >= competence.niveau:
                return {"possible": False, "raison": "L'élève maîtrise déjà cette compétence aussi bien que vous"}
                
        # Vérifier compétence d'enseignement (bonus)
        bonus_enseignement = 0
        if "enseignement" in self.personnage.competences:
            bonus_enseignement = self.personnage.competences["enseignement"].niveau
            
        # Calculer les chances de succès
        chance_base = 50 + (competence.niveau * 5) + (bonus_enseignement * 10)
        chance_base = min(95, chance_base)
        
        return {
            "possible": True,
            "chance_succes": chance_base,
            "duree_estimee": max(1, 10 - bonus_enseignement - (competence.niveau // 2))  # en heures
        }

# Système de rangs d'aventurier
class SystemeRangs:
    def __init__(self):
        self.criteres_promotion = {
            RangAventurier.F: {
                "niveau_min": 1,
                "quetes_completees": 0,
                "evaluation": 0
            },
            RangAventurier.E: {
                "niveau_min": 3,
                "quetes_completees": 3,
                "evaluation": 20
            },
            RangAventurier.D: {
                "niveau_min": 8,
                "quetes_completees": 10,
                "evaluation": 40
            },
            # Plus de rangs...
        }
    
    def verifier_eligibilite(self, personnage: 'Personnage', rang_cible: RangAventurier) -> Dict[str, Any]:
        """Vérifie si un personnage est éligible à une promotion de rang"""
        rang_actuel = personnage.rang_aventurier
        
        # Vérifier si le rang demandé est supérieur au rang actuel
        if rang_cible.value <= rang_actuel.value:
            return {"eligible": False, "raison": "Rang cible inférieur ou égal au rang actuel"}
            
        # Vérifier si le rang demandé est juste le rang suivant
        if rang_cible.value != rang_actuel.value + 1:
            return {"eligible": False, "raison": "Impossible de sauter des rangs"}
            
        # Récupérer les critères pour le rang cible
        criteres = self.criteres_promotion.get(rang_cible)
        if not criteres:
            return {"eligible": False, "raison": "Critères inconnus pour ce rang"}
            
        # Vérifier les critères
        resultats = {}
        
        # Niveau minimum
        niveau_ok = personnage.niveau >= criteres["niveau_min"]
        resultats["niveau"] = {
            "requis": criteres["niveau_min"],
            "actuel": personnage.niveau,
            "valide": niveau_ok
        }
        
        # Quêtes complétées
        quetes_ok = len(personnage.quetes_terminees) >= criteres["quetes_completees"]
        resultats["quetes"] = {
            "requis": criteres["quetes_completees"],
            "actuel": len(personnage.quetes_terminees),
            "valide": quetes_ok
        }
        
        # Évaluation (à implémenter selon le système)
        # Exemple simplifié
        evaluation = self._calculer_evaluation(personnage)
        evaluation_ok = evaluation >= criteres["evaluation"]
        resultats["evaluation"] = {
            "requis": criteres["evaluation"],
            "actuel": evaluation,
            "valide": evaluation_ok
        }
        
        # Tous les critères doivent être respectés
        eligible = niveau_ok and quetes_ok and evaluation_ok
        
        return {
            "eligible": eligible,
            "resultats": resultats,
            "rang_cible": rang_cible.name,
            "rang_actuel": rang_actuel.name
        }
        
    def _calculer_evaluation(self, personnage: 'Personnage') -> int:
        """Calcule le score d'évaluation d'un aventurier (à personnaliser)"""
        # Exemple simplifié
        score = 0
        
        # Points basés sur les relations avec les factions importantes
        for faction_id, reputation in personnage.reputation.items():
            if faction_id in ["guilde_aventuriers", "royaume_milis", "empire_asura"]:
                score += max(0, reputation) // 10
                
        # Points basés sur des accomplissements notables
        for evenement in personnage.historique_evenements:
            if evenement.get("type") == "monstre_dangereux_vaincu":
                score += 5
            elif evenement.get("type") == "quete_difficile_completee":
                score += 3
                
        # Points basés sur les compétences 
        for competence in personnage.competences.values():
            if competence.niveau >= 5:
                score += 2
                
        # Autres facteurs...
                
        return score
        
    def promouvoir(self, personnage: 'Personnage', rang_cible: RangAventurier) -> Dict[str, Any]:
        """Tente de promouvoir un personnage à un rang supérieur"""
        # Vérifier l'éligibilité
        verification = self.verifier_eligibilite(personnage, rang_cible)
        if not verification["eligible"]:
            return {"succes": False, "raison": verification.get("raison", "Non éligible")}
            
        # Appliquer la promotion
        ancien_rang = personnage.rang_aventurier
        personnage.rang_aventurier = rang_cible
        
        # Enregistrer l'événement
        personnage.historique_evenements.append({
            "type": "promotion_rang",
            "ancien_rang": ancien_rang.name,
            "nouveau_rang": rang_cible.name,
            "date": datetime.datetime.now().isoformat()
        })
        
        return {
            "succes": True,
            "message": f"Promotion au rang {rang_cible.name} réussie!",
            "ancien_rang": ancien_rang.name,
            "nouveau_rang": rang_cible.name
        }

# =================================================================
# === BLOC 10/12 : GESTION DE JEU ET SAUVEGARDE ===
# =================================================================

# Système de sauvegarde et chargement
class GestionnaireSauvegarde:
    def __init__(self, dossier_sauvegarde: str = "data"):
        self.dossier_sauvegarde = dossier_sauvegarde
        self.dossier_backups = "backups"
        self.extension = ".msave"
        os.makedirs(self.dossier_sauvegarde, exist_ok=True)
        os.makedirs(self.dossier_backups, exist_ok=True)
    
    def sauvegarder_jeu(self, etat_jeu: Dict[str, Any], nom_fichier: str = None) -> Dict[str, Any]:
        """Sauvegarde l'état actuel du jeu dans un fichier"""
        try:
            # Générer un nom si non fourni
            if not nom_fichier:
                horodatage = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                nom_personnage = etat_jeu.get("personnage", {}).get("nom", "inconnu")
                nom_fichier = f"{nom_personnage}_{horodatage}"
                
            # Ajouter l'extension si absente
            if not nom_fichier.endswith(self.extension):
                nom_fichier += self.extension
                
            # Chemin complet
            chemin_fichier = os.path.join(self.dossier_sauvegarde, nom_fichier)
            
            # Créer une copie de sauvegarde si le fichier existe déjà
            if os.path.exists(chemin_fichier):
                self._creer_backup(nom_fichier)
            
            # Ajouter des métadonnées
            etat_jeu["meta"] = {
                "version": VERSION,
                "date_sauvegarde": datetime.datetime.now().isoformat(),
                "nom_sauvegarde": nom_fichier
            }
            
            # Sérialiser et écrire
            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                json.dump(etat_jeu, f, cls=EncodeurJeuPerso, ensure_ascii=False, indent=2)
                
            logger.info(f"Jeu sauvegardé dans {chemin_fichier}")
            return {"succes": True, "fichier": chemin_fichier}
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return {"succes": False, "erreur": str(e)}
    
    def charger_jeu(self, nom_fichier: str) -> Dict[str, Any]:
        """Charge une sauvegarde depuis un fichier"""
        try:
            # Ajouter l'extension si absente
            if not nom_fichier.endswith(self.extension):
                nom_fichier += self.extension
                
            # Chemin complet
            chemin_fichier = os.path.join(self.dossier_sauvegarde, nom_fichier)
            
            # Vérifier que le fichier existe
            if not os.path.exists(chemin_fichier):
                return {"succes": False, "erreur": f"Fichier {chemin_fichier} introuvable"}
                
            # Lire et désérialiser
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                etat_jeu = json.load(f)
                
            # Vérifier la version
            version_sauvegarde = etat_jeu.get("meta", {}).get("version", "0.0")
            if version_sauvegarde != VERSION:
                logger.warning(f"Version de sauvegarde différente: {version_sauvegarde} vs {VERSION}")
                
            logger.info(f"Jeu chargé depuis {chemin_fichier}")
            return {"succes": True, "etat_jeu": etat_jeu}
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de format dans le fichier {nom_fichier}: {str(e)}")
            return {"succes": False, "erreur": f"Fichier corrompu: {str(e)}"}
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {str(e)}")
            return {"succes": False, "erreur": str(e)}
    
    def lister_sauvegardes(self) -> List[Dict[str, Any]]:
        """Liste toutes les sauvegardes disponibles avec leurs métadonnées"""
        try:
            sauvegardes = []
            
            for fichier in os.listdir(self.dossier_sauvegarde):
                if fichier.endswith(self.extension):
                    chemin = os.path.join(self.dossier_sauvegarde, fichier)
                    try:
                        # Extraire juste les métadonnées
                        with open(chemin, 'r', encoding='utf-8') as f:
                            donnees = json.load(f)
                            meta = donnees.get("meta", {})
                            personnage = donnees.get("personnage", {})
                            
                            sauvegardes.append({
                                "nom_fichier": fichier,
                                "date_sauvegarde": meta.get("date_sauvegarde", "Inconnue"),
                                "version": meta.get("version", "Inconnue"),
                                "nom_personnage": personnage.get("nom", "Inconnu"),
                                "niveau": personnage.get("niveau", 0),
                                "lieu": donnees.get("monde", {}).get("lieu_actuel", "Inconnu")
                            })
                    except Exception as e:
                        # En cas d'erreur, inclure quand même le fichier avec infos minimales
                        sauvegardes.append({
                            "nom_fichier": fichier,
                            "date_sauvegarde": "Erreur",
                            "erreur": str(e)
                        })
                        
            # Trier par date de sauvegarde (plus récent d'abord)
            sauvegardes.sort(key=lambda x: x.get("date_sauvegarde", ""), reverse=True)
            
            return sauvegardes
            
        except Exception as e:
            logger.error(f"Erreur lors du listage des sauvegardes: {str(e)}")
            return []
    
    def supprimer_sauvegarde(self, nom_fichier: str) -> Dict[str, Any]:
        """Supprime un fichier de sauvegarde"""
        try:
            # Ajouter l'extension si absente
            if not nom_fichier.endswith(self.extension):
                nom_fichier += self.extension
                
            # Chemin complet
            chemin_fichier = os.path.join(self.dossier_sauvegarde, nom_fichier)
            
            # Vérifier que le fichier existe
            if not os.path.exists(chemin_fichier):
                return {"succes": False, "erreur": f"Fichier {nom_fichier} introuvable"}
                
            # Créer une copie de sauvegarde avant suppression
            self._creer_backup(nom_fichier, "suppression")
            
            # Supprimer le fichier
            os.remove(chemin_fichier)
            
            logger.info(f"Sauvegarde {nom_fichier} supprimée")
            return {"succes": True}
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de {nom_fichier}: {str(e)}")
            return {"succes": False, "erreur": str(e)}
    
    def _creer_backup(self, nom_fichier: str, motif: str = "auto") -> bool:
        """Crée une copie de sauvegarde d'un fichier"""
        try:
            chemin_source = os.path.join(self.dossier_sauvegarde, nom_fichier)
            if not os.path.exists(chemin_source):
                return False
                
            # Générer le nom du backup
            horodatage = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            nom_base = nom_fichier.replace(self.extension, "")
            nom_backup = f"{nom_base}_{motif}_{horodatage}{self.extension}"
            
            chemin_destination = os.path.join(self.dossier_backups, nom_backup)
            
            # Copier le fichier
            shutil.copy2(chemin_source, chemin_destination)
            
            logger.info(f"Backup créé: {chemin_destination}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du backup: {str(e)}")
            return False

# Gestionnaire d'état de jeu
class GestionnaireJeu:
    def __init__(self):
        self.personnage = None  # Personnage du joueur
        self.monde = {}  # État du monde
        self.date_monde = DateMonde()  # Date dans l'univers du jeu
        self.emplacements = {}  # id -> Emplacement
        self.pnjs = {}  # id -> Personnage (PNJ)
        self.quetes = {}  # id -> Quete
        self.evenements = []  # Liste d'événements actifs
        self.factions = {}  # id -> Faction
        
        # Sous-systèmes
        self.gestionnaire_ia = None
        self.narrateur = None
        self.sauvegarde = GestionnaireSauvegarde()
        self.gestionnaire_dialogue = GestionnaireDialogue()
        
        # État de l'interface
        self.mode_actuel = "exploration"  # exploration, dialogue, combat, menu, etc.
        self.emplacement_actuel = None
        self.pnj_interlocuteur = None
        self.derniere_commande = ""
        self.derniere_reponse = ""
        
        # Historique de session
        self.historique_session = []
        self.debut_session = datetime.datetime.now()
        
    def initialiser_nouveau_jeu(self, config_personnage: Dict[str, Any]) -> Dict[str, Any]:
        """Initialise un nouveau jeu avec un personnage créé"""
        try:
            # Créer le personnage
            self.personnage = self._creer_personnage(config_personnage)
            
            # Initialiser le monde
            self._initialiser_monde()
            
            # Initialiser l'IA et le narrateur
            self.gestionnaire_ia = GestionnaireIA()
            self.narrateur = NarrateurDynamique(self.gestionnaire_ia)
            
            # Initialiser les systèmes de contexte
            self.gestionnaire_ia.initialiser_contexte(self.personnage, self.monde)
            
            # Enregistrer le début de l'aventure
            self.personnage.historique_evenements.append({
                "type": "debut_aventure",
                "date": datetime.datetime.now().isoformat(),
                "lieu": self.personnage.position_actuelle.get("lieu", "inconnu")
            })
            
            # Générer la narration d'introduction
            introduction = self._generer_introduction()
            
            return {
                "succes": True, 
                "personnage": self.personnage,
                "introduction": introduction
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du jeu: {str(e)}")
            return {"succes": False, "erreur": str(e)}
    
    def _creer_personnage(self, config: Dict[str, Any]) -> Personnage:
        """Crée un nouveau personnage basé sur la configuration fournie"""
        personnage_id = str(uuid.uuid4())
        
        # Créer le personnage de base
        personnage = Personnage(
            id=personnage_id,
            nom=config.get("nom", "Aventurier"),
            race=config.get("race", "Humain"),
            sexe=Sexe[config.get("sexe", "HOMME").upper()],
            age=config.get("age", 18),
            description=config.get("description", "")
        )
        
        # Définir les attributs de base
        for attribut in ["force", "vitesse", "endurance", "magie", "perception", 
                         "charisme", "chance", "intelligence", "volonte"]:
            if attribut in config:
                setattr(personnage, attribut, config[attribut])
        
        # Initialiser les affinités élémentaires
        elements_de_base = ["Eau", "Feu", "Terre", "Vent"]
        for element in elements_de_base:
            personnage.affinites_elementaires[element] = random.uniform(0.8, 1.2)
            
        # Élément ayant la plus grande affinité
        element_principal = max(personnage.affinites_elementaires.items(), key=lambda x: x[1])[0]
        personnage.affinites_elementaires[element_principal] += 0.3
        
        # Initialiser la position
        personnage.position_actuelle = {
            "region": "Royaume de Milis",
            "lieu": "Village de Buena",
            "coordonnees": (100, 200)
        }
        
        # Initialiser les compétences de base
        competences_base = {
            "survie": "Capacité à survivre en milieu hostile",
            "perception": "Capacité à repérer des détails",
            "premiers_soins": "Soins de base pour blessures légères"
        }
        
        for id_comp, desc in competences_base.items():
            personnage.ajouter_competence(id_comp, 1, desc)
            
        # Ajouter des compétences selon la race
        if personnage.race == "Humain":
            personnage.ajouter_competence("adaptabilité", 2, "Capacité à s'adapter à de nouvelles situations")
        elif personnage.race == "Elfe":
            personnage.ajouter_competence("perception_magique", 2, "Capacité à ressentir les flux magiques")
            personnage.magie += 2
        # Autres races...
            
        return personnage
    
    def _initialiser_monde(self):
        """Initialise l'état du monde pour un nouveau jeu"""
        # Définir la date de début
        self.date_monde = DateMonde(annee=407, mois=3, jour=15)
        
        # Initialiser les premières zones
        self._charger_emplacements()
        
        # Initialiser les PNJs de base
        self._charger_pnjs()
        
        # Initialiser les factions
        self._charger_factions()
        
        # Initialiser les quêtes de départ
        self._initialiser_quetes_depart()
        
        # Définir l'emplacement actuel
        lieu_depart_id = "buena_village"
        if lieu_depart_id in self.emplacements:
            self.emplacement_actuel = self.emplacements[lieu_depart_id]
            
        # État global du monde
        self.monde = {
            "date_actuelle": str(self.date_monde),
            "lieu_actuel": self.emplacement_actuel.nom if self.emplacement_actuel else "inconnu",
            "tensions_politiques": {
                "milis_asura": 65,  # Sur 100, tension entre royaumes
                "humains_demons": 80
            },
            "evenements_majeurs": []
        }
    
    def _charger_emplacements(self):
        """Charge les emplacements initiaux"""
        # Créer quelques lieux de base pour le début du jeu
        self.emplacements = {
            "buena_village": Emplacement(
                id="buena_village",
                nom="Village de Buena",
                type="village",
                region="Royaume de Milis",
                continent="Continent Central",
                description="Un petit village paisible niché entre des collines verdoyantes. Connu pour son agriculture et son atmosphère sereine.",
                danger=1,
                coordonnees=(100, 200),
                connecte_a=["route_milis_ouest", "foret_verte"],
                services={
                    "auberge": True,
                    "forge": True,
                    "magasin_general": True,
                    "temple": True
                }
            ),
            "route_milis_ouest": Emplacement(
                id="route_milis_ouest",
                nom="Route Ouest de Milis",
                type="route",
                region="Royaume de Milis",
                continent="Continent Central",
                description="Une route commerciale bien entretenue, reliant les villages de l'ouest à la capitale du royaume.",
                danger=2,
                coordonnees=(80, 200),
                connecte_a=["buena_village", "carrefour_marchands"]
            ),
            "foret_verte": Emplacement(
                id="foret_verte",
                nom="Forêt Verte",
                type="foret",
                region="Royaume de Milis",
                continent="Continent Central",
                description="Une forêt dense aux arbres majestueux. On raconte que des créatures magiques y vivent.",
                danger=4,
                coordonnees=(110, 210),
                connecte_a=["buena_village", "clairiere_mystique"],
                ressources={
                    "herbes": 20,
                    "bois": 50,
                    "fruits": 15
                }
            ),
            # Plus d'emplacements...
        }
        
        # Ajouter quelques habitants par défaut
        for pnj_id in ["marchand_buena", "aubergiste_buena", "garde_buena"]:
            self.emplacements["buena_village"].habitants.append(pnj_id)
    
    def _charger_pnjs(self):
        """Charge les PNJs initiaux"""
        self.pnjs = {
            "marchand_buena": Personnage(
                id="marchand_buena",
                nom="Geralt",
                titre="Marchand",
                race="Humain",
                sexe=Sexe.HOMME,
                age=45,
                description="Un marchand corpulent au sourire facile. Il connaît tous les potins du village.",
                niveau=3,
                est_canonique=False
            ),
            "aubergiste_buena": Personnage(
                id="aubergiste_buena",
                nom="Marian",
                titre="Aubergiste",
                race="Humain",
                sexe=Sexe.FEMME,
                age=38,
                description="Une femme énergique aux cheveux roux. Son auberge est réputée pour sa cuisine.",
                niveau=2,
                est_canonique=False
            ),
            "garde_buena": Personnage(
                id="garde_buena",
                nom="Terrence",
                titre="Garde du village",
                race="Humain",
                sexe=Sexe.HOMME,
                age=30,
                description="Un homme robuste à l'air sévère. Ancien aventurier reconverti en garde.",
                niveau=5,
                est_canonique=False
            ),
            # Plus de PNJs...
        }
        
        # Ajouter des traits de personnalité
        self.pnjs["marchand_buena"].traits.primaires = {
            "bavard": 8,
            "cupide": 6,
            "jovial": 7
        }
        
        self.pnjs["aubergiste_buena"].traits.primaires = {
            "travailleur": 9,
            "curieux": 7,
            "protecteur": 6
        }
        
        self.pnjs["garde_buena"].traits.primaires = {
            "discipline": 8,
            "méfiant": 7,
            "loyal": 9
        }
    
    def _charger_factions(self):
        """Charge les factions initiales"""
        self.factions = {
            "royaume_milis": Faction(
                id="royaume_milis",
                nom="Royaume de Milis",
                description="Un royaume prospère basé sur une religion monothéiste. Connu pour ses chevaliers et sa stabilité.",
                influence=80,
                alignement=Alignement.LOYAL
            ),
            "empire_asura": Faction(
                id="empire_asura",
                nom="Saint Empire d'Asura",
                description="Un vaste empire avec une puissante noblesse. Centre politique du monde humain.",
                influence=90,
                alignement=Alignement.NEUTRE
            ),
            "guilde_aventuriers": Faction(
                id="guilde_aventuriers",
                nom="Guilde des Aventuriers",
                description="Organisation qui régule les activités des aventuriers et offre des quêtes.",
                influence=75,
                alignement=Alignement.NEUTRE
            ),
            # Plus de factions...
        }
        
        # Définir les relations entre factions
        self.factions["royaume_milis"].relations = {
            "empire_asura": 40,  # Relation positive mais tendue
            "guilde_aventuriers": 70  # Bonne relation
        }
        
        self.factions["empire_asura"].relations = {
            "royaume_milis": 45,
            "guilde_aventuriers": 60
        }
    
    def _initialiser_quetes_depart(self):
        """Initialise les quêtes disponibles au début du jeu"""
        self.quetes = {
            "premier_pas": Quete(
                id="premier_pas",
                titre="Premiers Pas d'Aventurier",
                description="Rendez-vous à la Guilde des Aventuriers de Buena pour vous inscrire officiellement.",
                type=TypeQuete.PRINCIPALE,
                difficulte=DifficulteQuete.FACILE,
                objectifs=[
                    ObjectifQuete(
                        description="Parler au représentant de la Guilde à Buena",
                        cible_id="representant_guilde_buena"
                    )
                ],
                recompenses={
                    "or": 50,
                    "experience": 100,
                    "reputation": {"guilde_aventuriers": 10}
                }
            ),
            "herbes_medicinales": Quete(
                id="herbes_medicinales",
                titre="Herbes Médicinales",
                description="L'apothicaire a besoin d'herbes médicinales de la Forêt Verte pour préparer des potions.",
                type=TypeQuete.SECONDAIRE,
                difficulte=DifficulteQuete.FACILE,
                donneur_id="apothicaire_buena",
                objectifs=[
                    ObjectifQuete(
                        description="Récolter des herbes médicinales dans la Forêt Verte",
                        quantite_cible=5,
                        emplacement_id="foret_verte",
                        type_objectif="collecte"
                    ),
                    ObjectifQuete(
                        description="Rapporter les herbes à l'apothicaire",
                        cible_id="apothicaire_buena"
                    )
                ],
                recompenses={
                    "or": 100,
                    "experience": 150,
                    "objet": "potion_soin_mineur"
                }
            ),
            # Plus de quêtes...
        }
        
        # Placer les quêtes dans les emplacements appropriés
        if "buena_village" in self.emplacements:
            self.emplacements["buena_village"].quetes_disponibles.append("premier_pas")
    
    def _generer_introduction(self) -> str:
        """Génère le texte d'introduction pour un nouveau jeu"""
        nom = self.personnage.nom
        race = self.personnage.race
        lieu = self.emplacement_actuel.nom if self.emplacement_actuel else "un lieu inconnu"
        
        prompt = f"""
        Génère une introduction captivante pour le début de l'aventure avec les éléments suivants:
        
        PERSONNAGE:
        - Nom: {nom}
        - Race: {race}
        - Âge: {self.personnage.age}
        
        CONTEXTE:
        - Le personnage débute son aventure à {lieu}, dans le Royaume de Milis
        - Nous sommes en l'an 407, une période relativement paisible mais pleine d'opportunités
        - Le personnage aspire à devenir un aventurier reconnu
        
        L'introduction doit:
        1. Plonger le joueur dans l'univers de Mushoku Tensei
        2. Établir un point de départ clair pour l'aventure
        3. Créer un sentiment d'immersion et de possibilités
        4. Ne pas mentionner Rudeus Greyrat, car le joueur incarne son propre personnage
        
        Le ton doit être évocateur et prometteur d'aventures.
        """
        
        contexte = {
            "type_narration": "introduction",
            "importance": 10
        }
        
        return self.gestionnaire_ia.generer_narration(prompt, contexte)
    
    def generer_etat_jeu(self) -> Dict[str, Any]:
        """Génère un dictionnaire contenant l'état complet du jeu pour sauvegarde"""
        return {
            "personnage": self.personnage,
            "date_monde": self.date_monde,
            "monde": self.monde,
            "emplacements": self.emplacements,
            "pnjs": self.pnjs,
            "quetes": self.quetes,
            "evenements": self.evenements,
            "factions": self.factions,
            "emplacement_actuel_id": self.emplacement_actuel.id if self.emplacement_actuel else None,
            "pnj_interlocuteur_id": self.pnj_interlocuteur.id if self.pnj_interlocuteur else None,
            "mode_actuel": self.mode_actuel
        }

# Gestionnaire d'interface utilisateur
class InterfaceUtilisateur:
    def __init__(self, gestionnaire_jeu: GestionnaireJeu):
        self.gestionnaire = gestionnaire_jeu
        self.commandes_disponibles = {}
        self.alias_commandes = {}
        self.historique_commandes = []
        self.contexte_aide = {}
        self.initialiser_commandes()
    
    def initialiser_commandes(self):
        """Initialise les commandes disponibles"""
        # Commandes générales
        self.commandes_disponibles.update({
            "aide": {
                "fonction": self.commande_aide,
                "description": "Affiche l'aide et les commandes disponibles",
                "syntaxe": "aide [commande]",
                "aliases": ["help", "?"]
            },
            "regarder": {
                "fonction": self.commande_regarder,
                "description": "Examine les alentours ou un élément spécifique",
                "syntaxe": "regarder [cible]",
                "aliases": ["look", "observer", "examiner"]
            },
            "aller": {
                "fonction": self.commande_aller,
                "description": "Se déplace vers un lieu connecté",
                "syntaxe": "aller [destination]",
                "aliases": ["go", "voyager", "déplacer"]
            },
            "parler": {
                "fonction": self.commande_parler,
                "description": "Engage une conversation avec un PNJ",
                "syntaxe": "parler [personnage]",
                "aliases": ["talk", "discuter", "dialoguer"]
            },
            "inventaire": {
                "fonction": self.commande_inventaire,
                "description": "Affiche le contenu de votre inventaire",
                "syntaxe": "inventaire",
                "aliases": ["inv", "sac", "items"]
            },
            "utiliser": {
                "fonction": self.commande_utiliser,
                "description": "Utilise un objet de l'inventaire",
                "syntaxe": "utiliser [objet] [cible?]",
                "aliases": ["use", "employer", "activer"]
            },
            "stats": {
                "fonction": self.commande_stats,
                "description": "Affiche les statistiques du personnage",
                "syntaxe": "stats",
                "aliases": ["caractéristiques", "attributs", "status"]
            },
            "quêtes": {
                "fonction": self.commande_quetes,
                "description": "Affiche les quêtes actives et disponibles",
                "syntaxe": "quêtes [id_quête?]",
                "aliases": ["quests", "missions", "objectifs"]
            },
            "temps": {
                "fonction": self.commande_temps,
                "description": "Affiche la date et l'heure actuelles",
                "syntaxe": "temps",
                "aliases": ["date", "heure", "calendrier"]
            },
            "attendre": {
                "fonction": self.commande_attendre,
                "description": "Fait passer le temps",
                "syntaxe": "attendre [heures]",
                "aliases": ["wait", "repos", "patienter"]
            },
            "sauvegarder": {
                "fonction": self.commande_sauvegarder,
                "description": "Sauvegarde la partie en cours",
                "syntaxe": "sauvegarder [nom?]",
                "aliases": ["save", "enregistrer"]
            },
            "charger": {
                "fonction": self.commande_charger,
                "description": "Charge une sauvegarde",
                "syntaxe": "charger [nom]",
                "aliases": ["load", "restaurer"]
            },
            "quitter": {
                "fonction": self.commande_quitter,
                "description": "Quitte le jeu",
                "syntaxe": "quitter",
                "aliases": ["exit", "bye", "fin"]
            }
        })
        
        # Générer les alias
        for cmd, info in self.commandes_disponibles.items():
            for alias in info.get("aliases", []):
                self.alias_commandes[alias] = cmd
    
    def traiter_commande(self, commande_brute: str) -> Dict[str, Any]:
        """Traite une commande entrée par l'utilisateur"""
        # Sauvegarder dans l'historique
        self.historique_commandes.append(commande_brute)
        
        # Analyser la commande
        mots = commande_brute.lower().strip().split()
        if not mots:
            return {"message": "Veuillez entrer une commande. Tapez 'aide' pour voir les commandes disponibles."}
            
        # Identifier la commande principale
        commande = mots[0]
        arguments = mots[1:] if len(mots) > 1 else []
        
        # Vérifier les alias
        if commande in self.alias_commandes:
            commande = self.alias_commandes[commande]
            
        # Exécuter la commande
        if commande in self.commandes_disponibles:
            return self.commandes_disponibles[commande]["fonction"](arguments)
        else:
            # Si la commande n'est pas reconnue, essayer l'IA
            return self._traiter_commande_libre(commande_brute)
    
    def _traiter_commande_libre(self, texte: str) -> Dict[str, Any]:
        """Traite une commande libre en utilisant l'IA"""
        if not self.gestionnaire.gestionnaire_ia:
            return {"message": "Commande non reconnue. Tapez 'aide' pour voir les commandes disponibles."}
        
        mode = self.gestionnaire.mode_actuel
        contexte_specifique = {
            "mode": mode,
            "lieu": self.gestionnaire.emplacement_actuel.nom if self.gestionnaire.emplacement_actuel else "inconnu",
            "interlocuteur": self.gestionnaire.pnj_interlocuteur.nom if self.gestionnaire.pnj_interlocuteur else None
        }
        
        prompt = f"""
        Le joueur a entré: "{texte}"
        
        Dans le contexte actuel ({mode}), interprète cette entrée et génère une réponse appropriée.
        
        Si c'est une action que le personnage peut réaliser, décris le résultat de cette action.
        Si c'est une question sur le monde ou le lore, réponds de façon informative.
        Si c'est une demande impossible ou incompréhensible, explique pourquoi ce n'est pas possible.
        """
        
        reponse = self.gestionnaire.gestionnaire_ia.generer_narration(prompt, contexte_specifique)
        
        return {
            "message": reponse,
            "type": "libre",
            "interpretee": True
        }
    
    # Implémentation des commandes
    def commande_aide(self, args: List[str]) -> Dict[str, Any]:
        """Affiche l'aide du jeu"""
        if not args:
            # Afficher l'aide générale
            categories = {
                "Navigation": ["regarder", "aller"],
                "Interaction": ["parler", "utiliser"],
                "Personnage": ["inventaire", "stats"],
                "Jeu": ["quêtes", "temps", "attendre"],
                "Système": ["sauvegarder", "charger", "quitter", "aide"]
            }
            
            message = "=== AIDE DU JEU ===\n\n"
            
            for categorie, commandes in categories.items():
                message += f"== {categorie} ==\n"
                for cmd in commandes:
                    if cmd in self.commandes_disponibles:
                        info = self.commandes_disponibles[cmd]
                        message += f"- {cmd}: {info['description']}\n"
                message += "\n"
                
            message += "Pour plus d'informations sur une commande, tapez: aide [commande]"
            
        else:
            # Afficher l'aide pour une commande spécifique
            commande = args[0].lower()
            
            # Vérifier les alias
            if commande in self.alias_commandes:
                commande = self.alias_commandes[commande]
                
            if commande in self.commandes_disponibles:
                info = self.commandes_disponibles[commande]
                message = f"=== Commande: {commande} ===\n"
                message += f"Description: {info['description']}\n"
                message += f"Syntaxe: {info['syntaxe']}\n"
                
                if info.get('aliases'):
                    message += f"Alias: {', '.join(info['aliases'])}\n"
                    
                if commande in self.contexte_aide:
                    message += f"\n{self.contexte_aide[commande]}"
            else:
                message = f"Commande '{args[0]}' inconnue. Tapez 'aide' sans arguments pour voir la liste des commandes."
                
        return {"message": message}
    
    def commande_regarder(self, args: List[str]) -> Dict[str, Any]:
        """Examine les alentours ou un élément spécifique"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_aller(self, args: List[str]) -> Dict[str, Any]:
        """Se déplace vers un lieu connecté"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_parler(self, args: List[str]) -> Dict[str, Any]:
        """Engage une conversation avec un PNJ"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_inventaire(self, args: List[str]) -> Dict[str, Any]:
        """Affiche le contenu de l'inventaire"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_utiliser(self, args: List[str]) -> Dict[str, Any]:
        """Utilise un objet de l'inventaire"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_stats(self, args: List[str]) -> Dict[str, Any]:
        """Affiche les statistiques du personnage"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_quetes(self, args: List[str]) -> Dict[str, Any]:
        """Affiche les quêtes actives et disponibles"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_temps(self, args: List[str]) -> Dict[str, Any]:
        """Affiche la date et l'heure actuelles"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_attendre(self, args: List[str]) -> Dict[str, Any]:
        """Fait passer le temps"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_sauvegarder(self, args: List[str]) -> Dict[str, Any]:
        """Sauvegarde la partie en cours"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_charger(self, args: List[str]) -> Dict[str, Any]:
        """Charge une sauvegarde"""
        # Fonctionnalité à implémenter
        pass
    
    def commande_quitter(self, args: List[str]) -> Dict[str, Any]:
        """Quitte le jeu"""
        # Fonctionnalité à implémenter
        pass

# =================================================================
# === BLOC 11/12 : ÉVÉNEMENTS ALÉATOIRES ET RENCONTRES ===
# =================================================================

# Enums pour les événements et rencontres
class TypeEvenement(Enum):
    METEO = auto()
    RENCONTRE = auto()
    COMBAT = auto()
    DECOUVERTE = auto()
    FESTIVAL = auto()
    CATASTROPHE = auto()
    MONDE = auto()
    PERSONNEL = auto()
    QUETE = auto()
    VOYAGE = auto()

class RareteEvenement(Enum):
    COMMUN = auto()
    RARE = auto()
    TRES_RARE = auto()
    UNIQUE = auto()
    LEGENDAIRE = auto()

class ContexteEvenement(Enum):
    EXPLORATION = auto()
    ROUTE = auto()
    VILLE = auto()
    DONJON = auto()
    REPOS = auto()
    QUETE = auto()
    NUIT = auto()
    JOUR = auto()
    SAISON = auto()

# Système de génération d'événements
@dataclass
class EvenementAleatoire:
    id: str
    titre: str
    description: str
    type: TypeEvenement
    rarete: RareteEvenement = RareteEvenement.COMMUN
    contextes: List[ContexteEvenement] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    consequences: Dict[str, Any] = field(default_factory=dict)
    choix: List[Dict[str, Any]] = field(default_factory=list)
    poids_base: int = 100  # Pondération de base pour la sélection
    
    # Pour les événements liés aux régions/lieux
    regions_compatibles: List[str] = field(default_factory=list)
    types_lieux_compatibles: List[str] = field(default_factory=list)
    
    # Pour les événements temporels
    periode_journee: Optional[str] = None  # matin, jour, soir, nuit
    saisons_compatibles: List[str] = field(default_factory=list)
    
    # Pour la continuité narrative
    peut_se_repeter: bool = True
    delai_repetition: int = 0  # en jours
    importance_narrative: int = 1  # 1-10, influence sur l'histoire
    evenements_suivants: List[str] = field(default_factory=list)
    
    def verifier_conditions(self, personnage: 'Personnage', contexte: Dict[str, Any]) -> bool:
        """Vérifie si les conditions de déclenchement sont remplies"""
        for condition, valeur in self.conditions.items():
            # Vérification de niveau
            if condition == "niveau_min" and personnage.niveau < valeur:
                return False
                
            # Vérification de position
            elif condition == "region" and contexte.get("region") != valeur:
                return False
                
            # Vérification de quêtes
            elif condition.startswith("quete_"):
                id_quete = condition.replace("quete_", "")
                if condition.endswith("_active"):
                    if id_quete not in personnage.quetes_actives:
                        return False
                elif condition.endswith("_terminee"):
                    if id_quete not in personnage.quetes_terminees:
                        return False
                        
            # Vérification d'événements précédents
            elif condition.startswith("evenement_"):
                id_evt = condition.replace("evenement_", "")
                if id_evt not in contexte.get("evenements_passes", []):
                    return False
                    
            # Vérification d'attributs
            elif hasattr(personnage, condition) and getattr(personnage, condition) < valeur:
                return False
                
            # Vérification de période
            elif condition == "periode_journee" and contexte.get("periode_journee") != valeur:
                return False
                
            # Vérification de météo
            elif condition == "meteo" and contexte.get("meteo", {}).get("condition", "") != valeur:
                return False
                
        return True
    
    def appliquer_consequences(self, personnage: 'Personnage', choix_id: str, monde: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les conséquences d'un choix particulier"""
        # Trouver le choix sélectionné
        choix = next((c for c in self.choix if c.get("id") == choix_id), None)
        if not choix:
            return {"succes": False, "message": "Choix non valide"}
            
        consequences = choix.get("consequences", {})
        resultats = {}
        
        # Appliquer les conséquences
        for cle, valeur in consequences.items():
            # Modification d'attribut
            if hasattr(personnage, cle) and isinstance(getattr(personnage, cle), (int, float)):
                attribut_avant = getattr(personnage, cle)
                setattr(personnage, cle, attribut_avant + valeur)
                resultats[cle] = {"avant": attribut_avant, "apres": getattr(personnage, cle)}
                
            # Modification de ressources
            elif cle == "or":
                or_avant = personnage.inventaire.or_possede
                personnage.inventaire.or_possede += valeur
                resultats["or"] = {"avant": or_avant, "apres": personnage.inventaire.or_possede}
                
            # Gain d'objets
            elif cle == "objet" and isinstance(valeur, dict):
                id_objet = valeur.get("id")
                quantite = valeur.get("quantite", 1)
                # Logique pour ajouter l'objet à l'inventaire
                resultats["objet"] = {"id": id_objet, "quantite": quantite}
                
            # Gain d'expérience
            elif cle == "experience":
                niveau_avant = personnage.niveau
                exp_avant = personnage.experience
                # Utiliser le système de progression
                # personnage.progression.ajouter_experience(valeur)
                resultats["experience"] = {
                    "gain": valeur,
                    "avant": exp_avant,
                    "apres": personnage.experience,
                    "niveau_avant": niveau_avant,
                    "niveau_apres": personnage.niveau
                }
                
            # Modification de relation avec faction
            elif cle.startswith("faction_"):
                faction_id = cle.replace("faction_", "")
                if faction_id in personnage.reputation.factions:
                    rep_avant = personnage.reputation.factions[faction_id]
                    personnage.reputation.modifier_reputation("faction", faction_id, valeur)
                    resultats[cle] = {"avant": rep_avant, "apres": personnage.reputation.factions[faction_id]}
                    
            # Effets sur le monde
            elif cle.startswith("monde_"):
                attribut = cle.replace("monde_", "")
                if attribut in monde:
                    monde[attribut] = valeur
                    resultats[cle] = {"nouvelle_valeur": valeur}
                    
            # Déclenchement de quêtes
            elif cle == "quete_debloquee":
                resultats["quete"] = {"id": valeur, "statut": "débloquée"}
                # Logique pour débloquer la quête
                
        # Enregistrer l'événement dans l'historique du personnage
        personnage.historique_evenements.append({
            "type": "evenement",
            "id": self.id,
            "titre": self.titre,
            "choix": choix_id,
            "date": datetime.datetime.now().isoformat()
        })
        
        return {"succes": True, "resultats": resultats}

# Générateur d'événements aléatoires
class GenerateurEvenements:
    def __init__(self):
        self.evenements = {}  # id -> EvenementAleatoire
        self.evenements_par_type = defaultdict(list)  # type -> [ids]
        self.evenements_par_contexte = defaultdict(list)  # contexte -> [ids]
        self.evenements_par_region = defaultdict(list)  # region -> [ids]
        self.evenements_passes = []  # Liste des événements récemment déclenchés
        self.calendrier_evenements = {}  # date -> [ids] (événements programmés)
        
    def ajouter_evenement(self, evenement: EvenementAleatoire):
        """Ajoute un événement au générateur"""
        self.evenements[evenement.id] = evenement
        
        # Indexer par type
        self.evenements_par_type[evenement.type].append(evenement.id)
        
        # Indexer par contextes
        for contexte in evenement.contextes:
            self.evenements_par_contexte[contexte].append(evenement.id)
            
        # Indexer par région
        for region in evenement.regions_compatibles:
            self.evenements_par_region[region].append(evenement.id)
    
    def generer_evenement(self, personnage: 'Personnage', 
                        contexte_actuel: Dict[str, Any]) -> Optional[EvenementAleatoire]:
        """Génère un événement aléatoire adapté au contexte"""
        # Récupérer les contextes pertinents
        contextes = [ContexteEvenement[c.upper()] for c in contexte_actuel.get("contextes", [])
                    if c.upper() in ContexteEvenement.__members__]
        if not contextes:
            contextes = [ContexteEvenement.EXPLORATION]  # Contexte par défaut
            
        # Récupérer la région actuelle
        region = contexte_actuel.get("region", "")
        
        # Construction de la liste d'événements possibles
        evenements_possibles = []
        
        # Ajouter les événements compatibles avec le contexte
        for contexte in contextes:
            for evt_id in self.evenements_par_contexte[contexte]:
                evenement = self.evenements.get(evt_id)
                if not evenement:
                    continue
                    
                # Vérifier les autres conditions de compatibilité
                if (region and evenement.regions_compatibles and
                    region not in evenement.regions_compatibles):
                    continue
                    
                # Vérifier si l'événement peut se répéter
                if (not evenement.peut_se_repeter and
                    evt_id in [e.get("id") for e in self.evenements_passes]):
                    continue
                    
                # Vérifier les conditions spécifiques
                if not evenement.verifier_conditions(personnage, contexte_actuel):
                    continue
                    
                # Calculer le poids final
                poids = evenement.poids_base
                
                # Ajuster selon la rareté
                modificateurs_rarete = {
                    RareteEvenement.COMMUN: 1.0,
                    RareteEvenement.RARE: 0.5,
                    RareteEvenement.TRES_RARE: 0.2,
                    RareteEvenement.UNIQUE: 0.05,
                    RareteEvenement.LEGENDAIRE: 0.01
                }
                poids *= modificateurs_rarete.get(evenement.rarete, 1.0)
                
                # Ajouter à la liste avec son poids
                evenements_possibles.append((evenement, poids))
        
        # Si pas d'événements possibles
        if not evenements_possibles:
            return None
            
        # Sélection pondérée
        total_poids = sum(poids for _, poids in evenements_possibles)
        if total_poids <= 0:
            return None
            
        selection = random.uniform(0, total_poids)
        cumul = 0
        for evenement, poids in evenements_possibles:
            cumul += poids
            if selection <= cumul:
                # Enregistrer cet événement comme récemment déclenché
                self.evenements_passes.append({
                    "id": evenement.id,
                    "date": datetime.datetime.now().isoformat(),
                    "type": evenement.type.name
                })
                
                # Limiter la liste des événements récents
                if len(self.evenements_passes) > 20:
                    self.evenements_passes = self.evenements_passes[-20:]
                    
                return evenement
                
        # Si on arrive ici, retourner le dernier événement possible
        if evenements_possibles:
            evt, _ = evenements_possibles[-1]
            return evt
            
        return None
        
    def programmer_evenement(self, evenement_id: str, date_declenchement: DateMonde):
        """Programme un événement pour une date future spécifique"""
        if evenement_id not in self.evenements:
            logger.warning(f"Tentative de programmer un événement inexistant: {evenement_id}")
            return False
            
        date_str = str(date_declenchement)
        if date_str not in self.calendrier_evenements:
            self.calendrier_evenements[date_str] = []
            
        self.calendrier_evenements[date_str].append(evenement_id)
        logger.info(f"Événement {evenement_id} programmé pour le {date_str}")
        return True
        
    def verifier_evenements_programmes(self, date_actuelle: DateMonde) -> List[str]:
        """Vérifie s'il y a des événements programmés pour la date actuelle"""
        date_str = str(date_actuelle)
        return self.calendrier_evenements.get(date_str, [])
    
    def generer_consequences_narratives(self, evenement: EvenementAleatoire, choix: str) -> str:
        """Génère une description narrative des conséquences d'un événement"""
        # Cette fonction serait intégrée à l'IA pour générer un texte narratif
        # Selon l'événement et le choix fait
        pass

# Système de rencontres
class GenerateurRencontres:
    def __init__(self):
        self.rencontres = {}  # id -> dict avec infos de rencontre
        self.monstres = {}  # id -> info monstre
        self.pnjs_regionaux = defaultdict(list)  # région -> liste de PNJs typiques
        self.chances_rencontre = {
            # type de lieu -> probabilité de base pour 1h
            "foret": 0.15,
            "montagne": 0.12,
            "plaine": 0.08,
            "route": 0.05,
            "ville": 0.02,
            "donjon": 0.30
        }
    
    def initialiser_monstres(self):
        """Initialise la base de données des monstres"""
        # Exemples basés sur la faune de Mushoku Tensei
        self.monstres = {
            "loup_gris": {
                "nom": "Loup Gris",
                "type": "bête",
                "niveau": 3,
                "force": 15,
                "vitesse": 18,
                "endurance": 12,
                "danger": 2,
                "comportement": "meute",
                "environnements": ["foret", "plaine"],
                "loot": {
                    "fourrure_loup": {"chance": 0.7, "quantite": 1},
                    "croc_loup": {"chance": 0.4, "quantite": "1d2"}
                }
            },
            "gobelin": {
                "nom": "Gobelin",
                "type": "humanoïde",
                "niveau": 4,
                "force": 12,
                "vitesse": 14,
                "endurance": 10,
                "danger": 2,
                "comportement": "groupe",
                "environnements": ["foret", "grotte"],
                "loot": {
                    "oreille_gobelin": {"chance": 0.8, "quantite": 1},
                    "couteau_rouille": {"chance": 0.3, "quantite": 1}
                }
            },
            "araignee_venimeuse": {
                "nom": "Araignée Venimeuse",
                "type": "arthropode",
                "niveau": 5,
                "force": 14,
                "vitesse": 16,
                "endurance": 12,
                "danger": 3,
                "comportement": "solitaire",
                "environnements": ["foret", "grotte", "montagne"],
                "loot": {
                    "venin_araignee": {"chance": 0.5, "quantite": "1d3"},
                    "soie_araignee": {"chance": 0.6, "quantite": "1d4"}
                }
            },
            "troll_des_marais": {
                "nom": "Troll des Marais",
                "type": "géant",
                "niveau": 12,
                "force": 25,
                "vitesse": 10,
                "endurance": 30,
                "danger": 6,
                "comportement": "solitaire",
                "environnements": ["marais"],
                "loot": {
                    "peau_troll": {"chance": 0.9, "quantite": "1d3"},
                    "coeur_troll": {"chance": 0.3, "quantite": 1}
                }
            },
            # Plus de monstres...
        }
    
    def calculer_chance_rencontre(self, lieu: Emplacement, personnage: 'Personnage', 
                               duree_heures: float) -> float:
        """Calcule la probabilité d'une rencontre dans un lieu sur une période donnée"""
        # Obtenir la chance de base selon le type de lieu
        chance_base = self.chances_rencontre.get(lieu.type.lower(), 0.1)
        
        # Facteurs de modification
        # - Heure de la journée
        periode = personnage.position_actuelle.get("periode_journee", "jour")
        modif_periode = 1.5 if periode == "nuit" else 1.0
        
        # - Météo
        meteo = lieu.meteo_actuelle
        modif_meteo = 1.0
        if meteo.condition == ConditionMeteo.ORAGE:
            modif_meteo = 0.7  # Moins de rencontres pendant les orages
        elif meteo.condition == ConditionMeteo.BROUILLARD:
            modif_meteo = 1.3  # Plus de rencontres dans le brouillard
            
        # - Niveau de danger du lieu
        modif_danger = min(2.0, lieu.danger / 5)  # Maximum x2 pour les lieux très dangereux
        
        # - Compétence de furtivité du personnage
        modif_furtivite = 1.0
        if "furtivite" in personnage.competences:
            niveau_furtivite = personnage.competences["furtivite"].niveau
            modif_furtivite = max(0.5, 1.0 - (niveau_furtivite * 0.05))  # Maximum -50%
            
        # Calcul final, ajusté par la durée
        chance_finale = chance_base * modif_periode * modif_meteo * modif_danger * modif_furtivite
        chance_finale *= duree_heures  # Plus on reste longtemps, plus la chance augmente
        
        # Limiter à une valeur raisonnable
        return min(0.95, chance_finale)
    
    def generer_rencontre(self, lieu: Emplacement, personnage: 'Personnage', mode_force: bool = False) -> Dict[str, Any]:
        """Génère une rencontre aléatoire adaptée au lieu et au personnage"""
        # Déterminer si une rencontre se produit (sauf si forcé)
        if not mode_force:
            chance = self.calculer_chance_rencontre(lieu, personnage, 1.0)  # Pour 1 heure
            if random.random() > chance:
                return None  # Pas de rencontre
                
        # Déterminer le type de rencontre (hostile, neutre, amicale)
        type_distribution = {
            "hostile": 0.5,    # 50% hostile
            "neutre": 0.3,     # 30% neutre
            "amicale": 0.2     # 20% amicale
        }
        
        # Ajuster selon le lieu
        if lieu.type.lower() in ["ville", "village"]:
            type_distribution = {"hostile": 0.2, "neutre": 0.5, "amicale": 0.3}
        elif lieu.type.lower() in ["donjon", "ruine", "antre"]:
            type_distribution = {"hostile": 0.7, "neutre": 0.2, "amicale": 0.1}
            
        # Sélectionner le type
        type_rencontre = random.choices(
            list(type_distribution.keys()),
            weights=list(type_distribution.values())
        )[0]
        
        # Générer la rencontre selon le type
        if type_rencontre == "hostile":
            return self._generer_rencontre_hostile(lieu, personnage)
        elif type_rencontre == "neutre":
            return self._generer_rencontre_neutre(lieu, personnage)
        else:  # amicale
            return self._generer_rencontre_amicale(lieu, personnage)
    
    def _generer_rencontre_hostile(self, lieu: Emplacement, personnage: 'Personnage') -> Dict[str, Any]:
        """Génère une rencontre hostile (combat)"""
        # Déterminer les monstres adaptés à cet environnement
        monstres_possibles = []
        for id_monstre, info in self.monstres.items():
            if lieu.type.lower() in info["environnements"]:
                # Vérifier si le niveau est adapté
                niveau_monstre = info["niveau"]
                if abs(niveau_monstre - personnage.niveau) <= 5:  # Écart de niveau raisonnable
                    monstres_possibles.append((id_monstre, info))
        
        # Si pas de monstres adaptés, en prendre un au hasard
        if not monstres_possibles and self.monstres:
            monstres_possibles = random.sample(list(self.monstres.items()), min(3, len(self.monstres)))
        
        # Déterminer le nombre de monstres
        if not monstres_possibles:
            return None
            
        monstre_id, info_monstre = random.choice(monstres_possibles)
        
        # Le nombre dépend du comportement
        nombre = 1
        if info_monstre["comportement"] == "meute" or info_monstre["comportement"] == "groupe":
            max_groupe = 5
            # Ajuster selon l'écart de niveau
            diff_niveau = personnage.niveau - info_monstre["niveau"]
            if diff_niveau > 0:
                max_groupe += diff_niveau // 2
            nombre = random.randint(2, max_groupe)
        
        # Créer la rencontre
        rencontre = {
            "type": "hostile",
            "sous_type": "monstre",
            "lieu": lieu.nom,
            "monstres": [{
                "id": monstre_id,
                "nom": info_monstre["nom"],
                "niveau": info_monstre["niveau"],
                "nombre": nombre
            }],
            "difficulte_estimee": self._estimer_difficulte_combat(info_monstre, nombre, personnage),
            "experience_estimee": info_monstre["niveau"] * 15 * nombre,
            "loot_possible": info_monstre.get("loot", {})
        }
        
        return rencontre
    
    def _generer_rencontre_neutre(self, lieu: Emplacement, personnage: 'Personnage') -> Dict[str, Any]:
        """Génère une rencontre neutre (PNJ, voyageur, etc.)"""
        # Types de rencontres neutres possibles
        types_neutres = [
            "voyageur", "marchand", "chasseur", "chercheur_de_plantes",
            "aventurier", "groupe_de_voyageurs", "barde"
        ]
        
        # Ajuster selon le type de lieu
        if lieu.type.lower() in ["ville", "village"]:
            types_neutres.extend(["garde", "citadin", "artisan", "enfant"])
        elif lieu.type.lower() in ["foret", "montagne"]:
            types_neutres.extend(["ermite", "druide", "braconnier"])
        elif lieu.type.lower() in ["route"]:
            types_neutres.extend(["patrouille", "caravane", "messager"])
        
        sous_type = random.choice(types_neutres)
        
        # Niveau du PNJ - autour du niveau du joueur
        niveau_base = max(1, personnage.niveau + random.randint(-3, 3))
        
        # Génération du PNJ
        info_pnj = {
            "type": "neutre",
            "sous_type": sous_type,
            "nom": f"PNJ généré ({sous_type})",  # À remplacer par générateur de noms
            "niveau": niveau_base,
            "attitude": random.choice(["méfiant", "curieux", "indifférent", "poli", "amical"]),
            "commerce_possible": sous_type in ["marchand", "aventurier", "artisan"],
            "info_possible": True,  # Peut donner des informations
            "quete_possible": random.random() < 0.15  # 15% de chance d'avoir une quête
        }
        
        return {
            "type": "neutre",
            "lieu": lieu.nom,
            "pnj": info_pnj
        }
    
    def _generer_rencontre_amicale(self, lieu: Emplacement, personnage: 'Personnage') -> Dict[str, Any]:
        """Génère une rencontre amicale (aide, ressource, événement positif)"""
        # Types d'événements amicaux
        types_amicaux = [
            "ressource_rare", "voyageur_amical", "marchand_genereux", 
            "tresor_cache", "refuge", "aventurier_blesse"
        ]
        
        sous_type = random.choice(types_amicaux)
        
        # Contenu selon le type
        contenu = {}
        
        if sous_type == "ressource_rare":
            contenu = {
                "description": "Vous découvrez une ressource rare et précieuse.",
                "ressource": random.choice(["herbe_rare", "minerai_precieux", "cristal_magique"]),
                "quantite": random.randint(1, 3),
                "valeur": random.randint(50, 200)
            }
        elif sous_type == "tresor_cache":
            contenu = {
                "description": "Vous trouvez un petit trésor caché parmi les fourrés.",
                "or": random.randint(50, 300),
                "objets": [
                    {"id": "objet_aleatoire", "chance": 0.7},
                    {"id": "objet_rare", "chance": 0.3}
                ]
            }
        elif sous_type in ["voyageur_amical", "marchand_genereux", "aventurier_blesse"]:
            attitude = "amical" if sous_type != "aventurier_blesse" else "reconnaissant"
            contenu = {
                "pnj": {
                    "nom": f"PNJ généré ({sous_type})",  # À remplacer par générateur de noms
                    "niveau": max(1, personnage.niveau - 2),  # Un peu plus faible que le joueur
                    "attitude": attitude,
                    "commerce_possible": sous_type == "marchand_genereux",
                    "quete_possible": random.random() < 0.25,  # 25% de chance
                    "aide": sous_type != "aventurier_blesse",  # Peut aider sauf s'il est blessé
                    "besoin_aide": sous_type == "aventurier_blesse"  # A besoin d'aide s'il est blessé
                }
            }
        elif sous_type == "refuge":
            contenu = {
                "description": "Vous découvrez un refuge abandonné mais en bon état.",
                "abri": True,
                "securite": random.randint(60, 90),  # Sur 100
                "ressources": random.random() < 0.5,  # 50% de chance d'avoir des ressources
                "peut_se_reposer": True
            }
        
        return {
            "type": "amicale",
            "sous_type": sous_type,
            "lieu": lieu.nom,
            "contenu": contenu
        }
    
    def _estimer_difficulte_combat(self, info_monstre: Dict[str, Any], 
                                nombre: int, personnage: 'Personnage') -> int:
        """Estime la difficulté d'un combat sur une échelle de 1 à 10"""
        # Calculer la puissance du monstre
        puissance_monstre = (
            info_monstre["niveau"] * 10 +
            info_monstre["force"] + 
            info_monstre["vitesse"] + 
            info_monstre["endurance"]
        ) * nombre
        
        # Calculer la puissance du personnage
        puissance_perso = (
            personnage.niveau * 10 +
            personnage.force + 
            personnage.vitesse + 
            personnage.endurance
        )
        
        # Rapport de puissance
        rapport = puissance_monstre / max(1, puissance_perso)
        
        # Conversion en difficulté
        if rapport <= 0.5:
            return 1  # Très facile
        elif rapport <= 0.8:
            return 2  # Facile
        elif rapport <= 1.0:
            return 3  # Assez facile
        elif rapport <= 1.3:
            return 4  # Équilibré
        elif rapport <= 1.6:
            return 5  # Modéré
        elif rapport <= 2.0:
            return 6  # Difficile
        elif rapport <= 2.5:
            return 7  # Très difficile
        elif rapport <= 3.0:
            return 8  # Extrêmement difficile
        elif rapport <= 4.0:
            return 9  # Quasi impossible
        else:
            return 10  # Impossible

# Système d'ambiance et météo dynamique
class SystemeMeteo:
    def __init__(self):
        self.conditions_par_region = {}  # région -> conditions météo actuelles
        self.probabilites_transitions = {
            # condition actuelle -> {condition possible -> probabilité}
            ConditionMeteo.CLAIR: {
                ConditionMeteo.NUAGEUX: 0.2,
                ConditionMeteo.BROUILLARD: 0.05,
                ConditionMeteo.CLAIR: 0.75
            },
            ConditionMeteo.NUAGEUX: {
                ConditionMeteo.CLAIR: 0.3,
                ConditionMeteo.PLUIE: 0.2,
                ConditionMeteo.NUAGEUX: 0.5
            },
            # Autres transitions...
        }
        
        # Modificateurs de saison
        self.modificateurs_saison = {
            "printemps": {
                "temperature_base": 15,
                "plage_temperature": 10,  # ±10°C
                "conditions_favorisees": [ConditionMeteo.PLUIE, ConditionMeteo.NUAGEUX],
                "probabilite_precipitation": 0.3
            },
            "ete": {
                "temperature_base": 25,
                "plage_temperature": 8,
                "conditions_favorisees": [ConditionMeteo.CLAIR, ConditionMeteo.ORAGE],
                "probabilite_precipitation": 0.2
            },
            "automne": {
                "temperature_base": 15,
                "plage_temperature": 12,
                "conditions_favorisees": [ConditionMeteo.NUAGEUX, ConditionMeteo.BROUILLARD],
                "probabilite_precipitation": 0.4
            },
            "hiver": {
                "temperature_base": 0,
                "plage_temperature": 8,
                "conditions_favorisees": [ConditionMeteo.NEIGE, ConditionMeteo.CIEL_COUVERT],
                "probabilite_precipitation": 0.35
            }
        }
    
    def generer_meteo_initiale(self, region: str, saison: str, altitude: int = 0):
        """Génère des conditions météo initiales pour une région"""
        # Récupérer les modificateurs de saison
        mod_saison = self.modificateurs_saison.get(saison.lower(), self.modificateurs_saison["printemps"])
        
        # Générer la température de base
        temp_base = mod_saison["temperature_base"]
        variation = random.uniform(-mod_saison["plage_temperature"], mod_saison["plage_temperature"])
        temperature = temp_base + variation
        
        # Ajustement selon l'altitude
        temperature -= max(0, altitude // 100)  # -1°C tous les 100m
        
        # Déterminer la condition
        conditions_possibles = list(ConditionMeteo)
        conditions_favorisees = mod_saison.get("conditions_favorisees", [])
        
        # Favoriser certaines conditions selon la saison
        poids = [1.0 for _ in conditions_possibles]
        for i, condition in enumerate(conditions_possibles):
            if condition in conditions_favorisees:
                poids[i] = 3.0  # Trois fois plus probable
                
        # Sélectionner la condition
        condition = random.choices(conditions_possibles, weights=poids)[0]
        
        # Déterminer l'intensité
        intensite = random.randint(1, 10)
        
        # Créer l'objet météo
        meteo = ConditionsMeteo(
            condition=condition,
            temperature=round(temperature, 1),
            intensite=intensite,
            duree_prevue=random.randint(3, 12)  # En heures
        )
        
        # Sauvegarder pour la région
        self.conditions_par_region[region] = meteo
        
        return meteo
    
    def mettre_a_jour_meteo(self, region: str, saison: str, heures_ecoulees: int):
        """Met à jour les conditions météo d'une région après un temps écoulé"""
        # Récupérer la météo actuelle
        meteo_actuelle = self.conditions_par_region.get(region)
        
        # Créer une météo initiale si nécessaire
        if not meteo_actuelle:
            return self.generer_meteo_initiale(region, saison)
            
        # Vérifier si la météo doit changer
        if heures_ecoulees >= meteo_actuelle.duree_prevue:
            # Calculer la nouvelle condition
            nouvelle_condition = self._calculer_transition_meteo(meteo_actuelle.condition, saison)
            
            # Récupérer les modificateurs de saison
            mod_saison = self.modificateurs_saison.get(saison.lower(), self.modificateurs_saison["printemps"])
            
            # Calculer la nouvelle température
            variation_temp = random.uniform(-2, 2)  # Variation graduelle
            nouvelle_temp = meteo_actuelle.temperature + variation_temp
            
            # Ajustement vers la température de saison
            ecart_saison = mod_saison["temperature_base"] - nouvelle_temp
            nouvelle_temp += ecart_saison * 0.2  # Correction de 20% vers la température saisonnière
            
            # Nouvelle intensité
            nouvelle_intensite = max(1, min(10, meteo_actuelle.intensite + random.randint(-2, 2)))
            
            # Nouvelle durée prévue
            nouvelle_duree = random.randint(2, 8)  # Durée plus courte pour changements plus fréquents
            
            # Créer le nouvel objet météo
            nouvelle_meteo = ConditionsMeteo(
                condition=nouvelle_condition,
                temperature=round(nouvelle_temp, 1),
                intensite=nouvelle_intensite,
                duree_prevue=nouvelle_duree
            )
            
            # Sauvegarder
            self.conditions_par_region[region] = nouvelle_meteo
            
            return nouvelle_meteo
        else:
            # Météo inchangée, mais on peut faire varier légèrement la température
            meteo_actuelle.temperature += random.uniform(-0.5, 0.5)
            meteo_actuelle.temperature = round(meteo_actuelle.temperature, 1)
            
            # Décrémenter la durée prévue
            meteo_actuelle.duree_prevue -= heures_ecoulees
            
            return meteo_actuelle
    
    def _calculer_transition_meteo(self, condition_actuelle: ConditionMeteo, saison: str) -> ConditionMeteo:
        """Calcule la transition météorologique selon des probabilités et la saison"""
        # Récupérer les transitions possibles
        transitions = self.probabilites_transitions.get(
            condition_actuelle, 
            {c: 1/len(ConditionMeteo) for c in ConditionMeteo}  # Égalité par défaut
        )
        
        # Ajuster selon la saison
        mod_saison = self.modificateurs_saison.get(saison.lower(), self.modificateurs_saison["printemps"])
        conditions_favorisees = mod_saison.get("conditions_favorisees", [])
        
        transitions_ajustees = transitions.copy()
        for condition, probabilite in transitions_ajustees.items():
            if condition in conditions_favorisees:
                transitions_ajustees[condition] = probabilite * 1.5
        
        # Normaliser les probabilités
        total = sum(transitions_ajustees.values())
        transitions_normalisees = {k: v/total for k, v in transitions_ajustees.items()}
        
        # Sélectionner la nouvelle condition
        conditions = list(transitions_normalisees.keys())
        probas = list(transitions_normalisees.values())
        
        return random.choices(conditions, weights=probas)[0]

# Interface entre les événements et la narration
class GestionnaireEvenementsNarratifs:
    def __init__(self, gestionnaire_ia: GestionnaireIA, narrateur: NarrateurDynamique):
        self.generateur_evenements = GenerateurEvenements()
        self.generateur_rencontres = GenerateurRencontres()
        self.systeme_meteo = SystemeMeteo()
        self.ia = gestionnaire_ia
        self.narrateur = narrateur
        self.derniers_evenements = deque(maxlen=5)
    
    def initialiser_evenements_base(self):
        """Initialise les événements de base dans le système"""
        self.generateur_rencontres.initialiser_monstres()
        
        # Ajouter quelques événements de base
        evenement_rencontre_marchand = EvenementAleatoire(
            id="rencontre_marchand",
            titre="Marchand Itinérant",
            description="Vous croisez un marchand itinérant sur la route.",
            type=TypeEvenement.RENCONTRE,
            rarete=RareteEvenement.COMMUN,
            contextes=[ContexteEvenement.ROUTE, ContexteEvenement.EXPLORATION],
            choix=[
                {
                    "id": "commercer",
                    "texte": "Commercer avec le marchand",
                    "consequences": {
                        "rencontre_marchand": True
                    }
                },
                {
                    "id": "ignorer",
                    "texte": "Continuer votre chemin",
                    "consequences": {}
                },
                {
                    "id": "discuter",
                    "texte": "Engager la conversation pour obtenir des informations",
                    "consequences": {
                        "info_region": True
                    }
                }
            ]
        )
        self.generateur_evenements.ajouter_evenement(evenement_rencontre_marchand)
        
        # Événement météo
        evenement_orage = EvenementAleatoire(
            id="orage_violent",
            titre="Orage Violent",
            description="Un orage violent éclate subitement.",
            type=TypeEvenement.METEO,
            rarete=RareteEvenement.RARE,
            contextes=[ContexteEvenement.EXPLORATION, ContexteEvenement.ROUTE],
            conditions={
                "meteo": "ORAGE"
            },
            choix=[
                {
                    "id": "abri",
                    "texte": "Chercher un abri",
                    "consequences": {
                        "perte_temps": 2  # En heures
                    }
                },
                {
                    "id": "continuer",
                    "texte": "Continuer malgré l'orage",
                    "consequences": {
                        "endurance": -5,
                        "chance_maladie": 0.15
                    }
                }
            ]
        )
        self.generateur_evenements.ajouter_evenement(evenement_orage)
        
        # Événement découverte
        evenement_ruines = EvenementAleatoire(
            id="anciennes_ruines",
            titre="Anciennes Ruines",
            description="Vous découvrez des ruines anciennes partiellement enfouies.",
            type=TypeEvenement.DECOUVERTE,
            rarete=RareteEvenement.RARE,
            contextes=[ContexteEvenement.EXPLORATION],
            peut_se_repeter=False,
            choix=[
                {
                    "id": "explorer",
                    "texte": "Explorer les ruines",
                    "consequences": {
                        "quete_debloquee": "exploration_ruines"
                    }
                },
                {
                    "id": "noter",
                    "texte": "Noter l'emplacement et continuer",
                    "consequences": {
                        "decouverte_carte": "ruines_anciennes"
                    }
                },
                {
                    "id": "ignorer",
                    "texte": "Ignorer et passer votre chemin",
                    "consequences": {}
                }
            ]
        )
        self.generateur_evenements.ajouter_evenement(evenement_ruines)
        
        # Plus d'événements...
    
    def verifier_evenements(self, personnage: 'Personnage', contexte: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Vérifie s'il y a un événement à déclencher dans le contexte actuel"""
        # Récupérer le lieu
        lieu_id = personnage.position_actuelle.get("lieu_id")
        if not lieu_id:
            return None
            
        # Déterminer si un événement se produit
        # Probabilité de base: 10% par heure d'activité
        chance_evenement = 0.10
        
        # Facteurs qui augmentent la chance
        if "danger" in contexte:
            chance_evenement += contexte["danger"] * 0.02  # +2% par niveau de danger
            
        if "exploration" in contexte.get("activite", ""):
            chance_evenement += 0.05  # +5% en exploration
            
        if contexte.get("periode_journee") == "nuit":
            chance_evenement += 0.05  # +5% la nuit
            
        # Durée écoulée
        duree_heures = contexte.get("duree", 1.0)
        chance_finale = min(0.9, chance_evenement * duree_heures)
        
        # Décision aléatoire
        if random.random() > chance_finale and not contexte.get("forcer_evenement"):
            return None
            
        # Choisir entre événement et rencontre
        if random.random() < 0.6:  # 60% événements, 40% rencontres
            # Générer un événement
            evenement = self.generateur_evenements.generer_evenement(personnage, contexte)
            if evenement:
                return self._formater_evenement(evenement, personnage, contexte)
        else:
            # Générer une rencontre
            lieu = contexte.get("lieu")
            if lieu:
                rencontre = self.generateur_rencontres.generer_rencontre(lieu, personnage)
                if rencontre:
                    return self._formater_rencontre(rencontre, personnage, contexte)
                    
        return None
    
    def _formater_evenement(self, evenement: EvenementAleatoire, 
                          personnage: 'Personnage', contexte: Dict[str, Any]) -> Dict[str, Any]:
        """Formate un événement pour présentation au joueur"""
        # Générer la narration de l'événement
        narration = self.generer_narration_evenement(evenement, personnage, contexte)
        
        # Préparer les choix à présenter
        choix = []
        for option in evenement.choix:
            choix_formate = {
                "id": option["id"],
                "texte": option["texte"]
            }
            
            # Ajouter une indication si un choix nécessite une compétence
            if "competence_requise" in option:
                comp_id = option["competence_requise"]["id"]
                niveau_req = option["competence_requise"]["niveau"]
                
                a_competence = (comp_id in personnage.competences and 
                              personnage.competences[comp_id].niveau >= niveau_req)
                
                choix_formate["necessite_competence"] = comp_id
                choix_formate["niveau_requis"] = niveau_req
                choix_formate["competence_disponible"] = a_competence
                
            choix.append(choix_formate)
        
        # Enregistrer cet événement
        self.derniers_evenements.append({
            "type": "evenement",
            "id": evenement.id,
            "titre": evenement.titre,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Retourner l'événement formaté
        return {
            "type": "evenement",
            "evenement": evenement,
            "narration": narration,
            "choix": choix
        }
    
    def _formater_rencontre(self, rencontre: Dict[str, Any], 
                          personnage: 'Personnage', contexte: Dict[str, Any]) -> Dict[str, Any]:
        """Formate une rencontre pour présentation au joueur"""
        # Générer la narration de la rencontre
        narration = self.generer_narration_rencontre(rencontre, personnage, contexte)
        
        # Préparer les choix selon le type de rencontre
        choix = []
        
        if rencontre["type"] == "hostile":
            choix = [
                {"id": "combattre", "texte": "Se préparer au combat"},
                {"id": "fuir", "texte": "Tenter de fuir"},
                {"id": "intimider", "texte": "Tenter d'intimider"}
            ]
            
            # Ajouter des options spéciales si compétences disponibles
            if "furtivite" in personnage.competences and personnage.competences["furtivite"].niveau >= 3:
                choix.append({"id": "embuscade", "texte": "Préparer une embuscade"})
                
            if "persuasion" in personnage.competences and personnage.competences["persuasion"].niveau >= 4:
                choix.append({"id": "negocier", "texte": "Tenter de négocier"})
                
        elif rencontre["type"] == "neutre":
            choix = [
                {"id": "saluer", "texte": "Saluer poliment"},
                {"id": "ignorer", "texte": "Ignorer et passer votre chemin"}
            ]
            
            if rencontre["pnj"].get("commerce_possible"):
                choix.append({"id": "commercer", "texte": "Proposer du commerce"})
                
            if rencontre["pnj"].get("info_possible"):
                choix.append({"id": "demander_info", "texte": "Demander des informations"})
                
        elif rencontre["type"] == "amicale":
            choix = [
                {"id": "accepter", "texte": "Accepter l'aide/l'opportunité"},
                {"id": "decliner", "texte": "Décliner poliment"}
            ]
            
            sous_type = rencontre.get("sous_type")
            if sous_type == "aventurier_blesse":
                choix = [
                    {"id": "aider", "texte": "Aider l'aventurier blessé"},
                    {"id": "ignorer", "texte": "Passer votre chemin"}
                ]
            elif sous_type == "ressource_rare" or sous_type == "tresor_cache":
                choix = [
                    {"id": "prendre", "texte": "Prendre la ressource/le trésor"},
                    {"id": "examiner", "texte": "Examiner attentivement avant de prendre"},
                    {"id": "laisser", "texte": "Laisser en place"}
                ]
        
        # Enregistrer cette rencontre
        self.derniers_evenements.append({
            "type": "rencontre",
            "sous_type": rencontre["type"],
            "lieu": rencontre["lieu"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Retourner la rencontre formatée
        return {
            "type": "rencontre",
            "rencontre": rencontre,
            "narration": narration,
            "choix": choix
        }
    
    def generer_narration_evenement(self, evenement: EvenementAleatoire, 
                                 personnage: 'Personnage', contexte: Dict[str, Any]) -> str:
        """Génère une description narrative d'un événement"""
        lieu = contexte.get("lieu", {}).get("nom", "")
        meteo = contexte.get("meteo", {}).get("condition", "").lower()
        periode = contexte.get("periode_journee", "jour")
        
        prompt = f"""
        Décris cet événement de façon immersive, en tenant compte du contexte:
        
        ÉVÉNEMENT: {evenement.titre}
        TYPE: {evenement.type.name}
        DESCRIPTION BASE: {evenement.description}
        
        CONTEXTE:
        - Lieu: {lieu}
        - Météo: {meteo}
        - Moment de la journée: {periode}
        
        Adapte la description pour qu'elle soit cohérente avec le contexte actuel.
        La narration doit être immersive et captivante, sans répéter mécaniquement les informations ci-dessus.
        N'inclus pas les choix possibles dans ta description.
        """
        
        contexte_ia = {
            "type_narration": "evenement",
            "importance": 7 if evenement.rarete.value > RareteEvenement.COMMUN.value else 5
        }
        
        return self.ia.generer_narration(prompt, contexte_ia)
    
    def generer_narration_rencontre(self, rencontre: Dict[str, Any], 
                                 personnage: 'Personnage', contexte: Dict[str, Any]) -> str:
        """Génère une description narrative d'une rencontre"""
        lieu = contexte.get("lieu", {}).get("nom", "")
        type_rencontre = rencontre["type"]
        details = ""
        
        if type_rencontre == "hostile" and "monstres" in rencontre:
            monstre = rencontre["monstres"][0]
            details = f"Monstre: {monstre['nom']} (x{monstre['nombre']})"
        elif type_rencontre == "neutre" and "pnj" in rencontre:
            pnj = rencontre["pnj"]
            details = f"PNJ: {pnj['nom']} ({pnj['sous_type']}), attitude: {pnj['attitude']}"
        elif type_rencontre == "amicale":
            details = f"Type: {rencontre.get('sous_type', 'rencontre amicale')}"
        
        prompt = f"""
        Décris cette rencontre de façon immersive et captivante:
        
        TYPE DE RENCONTRE: {type_rencontre}
        LIEU: {lieu}
        DÉTAILS: {details}
        
        CONTEXTE:
        - Période: {contexte.get('periode_journee', 'jour')}
        - Météo: {contexte.get('meteo', {}).get('condition', '').lower()}
        
        Pour une rencontre hostile, mets l'accent sur la tension et le danger.
        Pour une rencontre neutre, décris l'apparence et l'attitude du PNJ.
        Pour une rencontre amicale, souligne l'opportunité ou l'aspect positif.
        
        La description doit être immersive et évocatrice, adaptée au contexte.
        N'inclus pas les choix possibles dans ta description.
        """
        
        contexte_ia = {
            "type_narration": "rencontre",
            "importance": 8 if type_rencontre == "hostile" else 6
        }
        
        return self.ia.generer_narration(prompt, contexte_ia)
    
    def gerer_reponse_evenement(self, evenement: EvenementAleatoire, 
                             choix_id: str, personnage: 'Personnage', monde: Dict[str, Any]) -> Dict[str, Any]:
        """Gère la réponse du joueur à un événement"""
        # Appliquer les conséquences du choix
        resultat_choix = evenement.appliquer_consequences(personnage, choix_id, monde)
        
        # Générer la narration du résultat
        narration = self.generer_narration_consequence(evenement, choix_id, resultat_choix, personnage)
        
        return {
            "succes": resultat_choix["succes"],
            "narration": narration,
            "resultats": resultat_choix.get("resultats", {})
        }
    
    def generer_narration_consequence(self, evenement: EvenementAleatoire, 
                                    choix_id: str, resultat: Dict[str, Any], 
                                    personnage: 'Personnage') -> str:
        """Génère une narration pour les conséquences d'un choix"""
        # Trouver le choix effectué
        choix = next((c for c in evenement.choix if c.get("id") == choix_id), None)
        if not choix:
            return "Les conséquences de votre décision restent floues."
            
        # Récupérer les conséquences pour le prompt
        consequences = []
        for cle, valeur in resultat.get("resultats", {}).items():
            if isinstance(valeur, dict):
                if "avant" in valeur and "apres" in valeur:
                    delta = valeur["apres"] - valeur["avant"]
                    direction = "augmente de" if delta > 0 else "diminue de"
                    consequences.append(f"{cle} {direction} {abs(delta)}")
                elif "gain" in valeur:
                    consequences.append(f"Gain de {valeur['gain']} points d'expérience")
                elif "id" in valeur and "quantite" in valeur:
                    consequences.append(f"Obtention de {valeur['quantite']} {valeur['id']}")
            elif isinstance(valeur, str):
                consequences.append(f"{cle}: {valeur}")
                
        consequences_texte = "\n".join(["- " + c for c in consequences]) if consequences else "Aucun changement majeur."
        
        prompt = f"""
        Décris les conséquences de la décision prise face à cet événement:
        
        ÉVÉNEMENT: {evenement.titre}
        DÉCISION PRISE: {choix.get('texte', 'Décision inconnue')}
        
        CONSÉQUENCES:
        {consequences_texte}
        
        Transforme ces informations techniques en une narration fluide et immersive.
        Décris comment la situation évolue suite à cette décision, et comment {personnage.nom} en est affecté.
        Sois expressif et évocateur, sans être trop long ou verbeux.
        """
        
        contexte_ia = {
            "type_narration": "consequence",
            "importance": 6
        }
        
        return self.ia.generer_narration(prompt, contexte_ia)


# ===========================================
# === BLOC 13/13 : INTERFACE PRINCIPALE ====
# ===========================================

import os
import sys
import time
import random
import uuid  # Pour générer des ID uniques
from enum import Enum
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mushoku_tensei_rp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mushoku_tensei_rp")

# Version du jeu
VERSION = "3.0"

# Modèle IA par défaut
DEFAULT_MODEL = "mythomax:latest"

# Liste des races disponibles
RACES = ["Humain", "Elfe", "Nain", "Demi-Elfe", "Demi-Démon", "Bête-Humain", "Démon"]

class Couleur:
    """Codes ANSI pour colorer le texte dans le terminal"""
    RESET = '\033[0m'
    GRAS = '\033[1m'
    SOULIGNÉ = '\033[4m'
    NOIR = '\033[30m'
    ROUGE = '\033[31m'
    VERT = '\033[32m'
    JAUNE = '\033[33m'
    BLEU = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BLANC = '\033[37m'
    FOND_NOIR = '\033[40m'
    FOND_ROUGE = '\033[41m'
    FOND_VERT = '\033[42m'
    FOND_JAUNE = '\033[43m'
    FOND_BLEU = '\033[44m'
    FOND_MAGENTA = '\033[45m'
    FOND_CYAN = '\033[46m'
    FOND_BLANC = '\033[47m'

# États du jeu
class ÉtatJeu(Enum):
    MENU_PRINCIPAL = 0
    CRÉATION_PERSONNAGE = 1
    JEU_ACTIF = 2
    OPTIONS = 3
    QUITTER = 4

# Configuration du jeu
class ConfigJeu:
    def __init__(self):
        self.modèle_ia = DEFAULT_MODEL
        self.personnage_actuel = None
        self.historique = []
        self.état = ÉtatJeu.MENU_PRINCIPAL
        self.dernière_sauvegarde = None
        self.mode_verbose = False
        self.mode_debug = False
        
config_jeu = ConfigJeu()

# Client IA pour interagir avec Ollama - Version qui ne dépend pas de l'importation externe
class ClientIA:
    def chat_completion(self, model, messages, max_tokens=None):
        """Version simplifiée qui fonctionne sans dépendance externe"""
        logger.info(f"Appel à l'IA avec le modèle {model}")
        
        # Simuler une réponse basée sur des mots clés dans le dernier message
        dernier_message = messages[-1]["content"] if messages else ""
        
        # Réponses simulées en fonction du contexte
        if "naissance" in dernier_message.lower():
            reponse = "Tu ouvres les yeux pour la première fois. Le monde qui t'entoure est flou, rempli de formes et de couleurs indistinctes. Des voix douces murmurent autour de toi, et tu sens une chaleur réconfortante contre ta peau."
        elif "explorer" in dernier_message.lower():
            reponse = "Tu explores les environs avec curiosité. Les bâtiments autour de toi témoignent d'une architecture ancienne, mêlant pierre et bois. Au loin, les montagnes se dressent majestueusement, leurs sommets couverts de neige étincelant sous le soleil."
        elif "parler" in dernier_message.lower():
            reponse = "En t'approchant pour engager la conversation, tu remarques que ton interlocuteur t'observe avec un mélange de curiosité et de méfiance. Après quelques instants d'hésitation, il t'adresse un signe de tête poli, t'invitant à parler."
        else:
            reponse = "Le monde de Mushoku Tensei s'étend devant toi, plein de mystères à découvrir et d'aventures à vivre. Que souhaites-tu faire maintenant?"
        
        return {"response": reponse}

# Initialiser le client IA
client = ClientIA()

# Classe Personnage
class Personnage:
    """Classe représentant un personnage joueur"""
    def __init__(self, id, nom, race):
        self.id = id
        self.nom = nom
        self.race = race

# Fonctions d'affichage
def effacer_écran():
    """Efface l'écran du terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimer_titre():
    """Affiche le titre du jeu"""
    effacer_écran()
    print(f"{Couleur.CYAN}{Couleur.GRAS}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                                                           ║")
    print("║              MUSHOKU TENSEI ROLEPLAY AGENT                ║")
    print("║                  AVENTURE IMMERSIVE v3.0                  ║")
    print("║                                                           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Couleur.RESET}")

def imprimer_séparateur():
    """Affiche un séparateur dans le terminal"""
    print(f"{Couleur.CYAN}{'─' * 60}{Couleur.RESET}")

def imprimer_message(texte, couleur=Couleur.BLANC, délai=0.03):
    """Affiche un texte caractère par caractère pour un effet d'écriture"""
    for caractère in texte:
        print(f"{couleur}{caractère}{Couleur.RESET}", end='', flush=True)
        time.sleep(délai)
    print()

def imprimer_narration(texte):
    """Affiche un texte de narration"""
    imprimer_message(texte, Couleur.JAUNE, 0.02)

def imprimer_dialogue(personnage, texte):
    """Affiche un dialogue avec le nom du personnage"""
    print(f"{Couleur.CYAN}{Couleur.GRAS}{personnage}{Couleur.RESET}: ", end='')
    imprimer_message(texte, Couleur.BLANC)

def imprimer_action(texte):
    """Affiche une action réalisée par le joueur"""
    print(f"{Couleur.VERT}» {texte}{Couleur.RESET}")

def imprimer_système(texte):
    """Affiche un message système"""
    print(f"{Couleur.MAGENTA}[Système] {texte}{Couleur.RESET}")

def imprimer_erreur(texte):
    """Affiche un message d'erreur"""
    print(f"{Couleur.ROUGE}[ERREUR] {texte}{Couleur.RESET}")

def afficher_menu_principal():
    """Affiche le menu principal du jeu"""
    imprimer_titre()
    print(f"{Couleur.CYAN}╔═══════════════════ MENU PRINCIPAL ═══════════════════╗{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} 1. Nouvelle Partie                                   {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} 2. Charger une Partie                                {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} 3. Options                                           {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} 4. À propos                                          {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} 5. Quitter                                           {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}╚═══════════════════════════════════════════════════════╝{Couleur.RESET}")

def afficher_menu_choix(titre, options, permettre_aleatoire=True):
    """Affiche un menu de choix et retourne l'option sélectionnée"""
    print(f"{Couleur.CYAN}{titre}:{Couleur.RESET}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    if permettre_aleatoire:
        print(f"0. Aléatoire")
    
    while True:
        choix = input(f"{Couleur.JAUNE}Votre choix {f'(0-{len(options)})' if permettre_aleatoire else f'(1-{len(options)})' }: {Couleur.RESET}")
        
        # Option aléatoire
        if choix == "0" and permettre_aleatoire:
            choix_index = random.randint(0, len(options) - 1)
            imprimer_système(f"Choix aléatoire : {options[choix_index]}")
            return options[choix_index]
        
        # Option spécifique
        try:
            choix_num = int(choix)
            if 1 <= choix_num <= len(options):
                return options[choix_num - 1]
            else:
                imprimer_erreur(f"Veuillez entrer un nombre entre {'0' if permettre_aleatoire else '1'} et {len(options)}.")
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")

def mettre_a_jour_personnalité(personnage, action, contexte, amplitude=5):
    """
    Met à jour les traits de personnalité du joueur en fonction de ses actions
    
    Parameters:
    -----------
    personnage : Personnage
        L'objet personnage du joueur
    action : str
        Description de l'action prise par le joueur
    contexte : str
        Le contexte dans lequel l'action a été prise
    amplitude : int
        L'importance de l'impact de l'action sur les traits (1-10)
    """
    # Si l'amplitude est en dehors des limites, on la normalise
    amplitude = max(1, min(10, amplitude))
    
    # Analyse de l'action et mise à jour des traits (cette partie pourrait utiliser l'IA)
    # Exemple simple basé sur des mots-clés
    modification = {}
    
    # Analyse du courage
    if any(mot in action.lower() for mot in ["attaquer", "défier", "affronter", "risquer"]):
        modification["courage"] = amplitude
    elif any(mot in action.lower() for mot in ["fuir", "éviter", "cacher", "reculer"]):
        modification["courage"] = -amplitude
    
    # Analyse de l'empathie
    if any(mot in action.lower() for mot in ["aider", "soigner", "sauver", "protéger"]):
        modification["empathie"] = amplitude
    elif any(mot in action.lower() for mot in ["ignorer", "abandonner", "trahir"]):
        modification["empathie"] = -amplitude
    
    # Analyse de l'ambition
    if any(mot in action.lower() for mot in ["conquérir", "réussir", "dominer", "gagner"]):
        modification["ambition"] = amplitude
    elif any(mot in action.lower() for mot in ["renoncer", "abandonner", "céder"]):
        modification["ambition"] = -amplitude
    
    # Analyse de la prudence
    if any(mot in action.lower() for mot in ["réfléchir", "planifier", "analyser", "observer"]):
        modification["prudence"] = amplitude
    elif any(mot in action.lower() for mot in ["foncer", "agir", "immédiatement", "spontanément"]):
        modification["prudence"] = -amplitude
        
    # Analyse de la sociabilité
    if any(mot in action.lower() for mot in ["parler", "discuter", "socialiser", "rencontrer"]):
        modification["sociabilite"] = amplitude
    elif any(mot in action.lower() for mot in ["isoler", "éviter", "seul", "silence"]):
        modification["sociabilite"] = -amplitude
        
    # Analyse de la moralité
    if any(mot in action.lower() for mot in ["honneur", "justice", "vérité", "bien"]):
        modification["moralite"] = amplitude
    elif any(mot in action.lower() for mot in ["voler", "mentir", "tromper", "tricher"]):
        modification["moralite"] = -amplitude
        
    # Analyse du tempérament
    if any(mot in action.lower() for mot in ["calme", "patient", "méditer", "apaiser"]):
        modification["temperament"] = -amplitude
    elif any(mot in action.lower() for mot in ["colère", "crier", "énerver", "s'emporter"]):
        modification["temperament"] = amplitude
    
    # Application des modifications avec limites (0-100)
    for trait, valeur in modification.items():
        if trait in personnage.traits_personnalite:
            ancien = personnage.traits_personnalite[trait]
            nouveau = max(0, min(100, ancien + valeur))
            personnage.traits_personnalite[trait] = nouveau
            
            # Si changement significatif, l'ajouter à l'historique des actions
            if abs(nouveau - ancien) >= 5:
                personnage.historique_actions.append({
                    "trait": trait,
                    "avant": ancien,
                    "après": nouveau,
                    "action": action,
                    "contexte": contexte
                })
                
                # Si changement très significatif, l'ajouter au journal
                if abs(nouveau - ancien) >= 10:
                    # Message adapté au trait et à la direction du changement
                    messages = {
                        "courage": {
                            "positif": "Tu te sens plus confiant face au danger.",
                            "négatif": "Tu te sens plus méfiant face au danger."
                        },
                        "empathie": {
                            "positif": "Tu ressens davantage les émotions des autres.",
                            "négatif": "Tu te sens plus détaché des émotions des autres."
                        },
                        "ambition": {
                            "positif": "Tu te sens plus déterminé à atteindre tes objectifs.",
                            "négatif": "Tu commences à te satisfaire de ta situation actuelle."
                        },
                        "prudence": {
                            "positif": "Tu prends l'habitude de réfléchir avant d'agir.",
                            "négatif": "Tu te surprends à agir plus spontanément."
                        },
                        "sociabilite": {
                            "positif": "Tu te sens plus à l'aise en compagnie des autres.",
                            "négatif": "Tu préfères de plus en plus la solitude."
                        },
                        "moralite": {
                            "positif": "Tes convictions morales se renforcent.",
                            "négatif": "Tu commences à voir le monde avec plus de pragmatisme."
                        },
                        "temperament": {
                            "positif": "Tu remarques que tu t'emportes plus facilement.",
                            "négatif": "Tu te sens plus calme face aux contrariétés."
                        }
                    }
                    
                    direction = "positif" if nouveau > ancien else "négatif"
                    if trait in messages and direction in messages[trait]:
                        personnage.journal.append({
                            "texte": messages[trait][direction],
                            "contexte": contexte
                        })

def traiter_commande(commande):
    """Traite une commande entrée par l'utilisateur"""
    # Commandes système
    if commande.startswith("/"):
        return traiter_commande_système(commande[1:])
    
    # Commandes de jeu normales
    historique_pour_ia = config_jeu.historique[-10:] if config_jeu.historique else []
    
    # Créer le contexte pour l'IA
    messages = [
        {"role": "system", "content": f"""Tu es le maître du jeu pour une aventure immersive dans 
        l'univers de Mushoku Tensei. Le joueur incarne {config_jeu.personnage_actuel.nom if config_jeu.personnage_actuel else 'un aventurier'}.
        Réponds de manière immersive, descriptive et détaillée. Mets l'accent sur l'ambiance, les sensations et les émotions.
        N'utilise jamais de métadonnées ou de balises techniques dans tes réponses.
        Adapte subtilement tes réponses aux traits de personnalité actuels du joueur sans les mentionner explicitement."""},
    ]
    
    # Ajouter l'historique récent
    for h in historique_pour_ia:
        messages.append({"role": h["role"], "content": h["contenu"]})
    
    # Ajouter la commande actuelle
    messages.append({"role": "user", "content": commande})
    
    try:
        # Appeler l'IA
        réponse = client.chat_completion(
            model=config_jeu.modèle_ia,
            messages=messages
        )
        
        contenu_réponse = ""
        if isinstance(réponse, dict):
            # Adapter selon la structure exacte retournée par notre client personnalisé
            if "response" in réponse and isinstance(réponse["response"], dict):
                contenu_réponse = réponse["response"].get("content", "")
            elif "response" in réponse and isinstance(réponse["response"], str):
                contenu_réponse = réponse["response"]
            elif "choices" in réponse:
                contenu_réponse = réponse["choices"][0]["message"]["content"]
            else:
                contenu_réponse = str(réponse)
        else:
            contenu_réponse = str(réponse)
        
        # Enregistrer dans l'historique
        config_jeu.historique.append({"role": "user", "contenu": commande})
        config_jeu.historique.append({"role": "assistant", "contenu": contenu_réponse})
        
        # Mettre à jour la personnalité en fonction de l'action
        if config_jeu.personnage_actuel and contenu_réponse:
            mettre_a_jour_personnalité(
                config_jeu.personnage_actuel, 
                commande,  # l'action du joueur
                contenu_réponse,  # le contexte/conséquence 
                amplitude=3  # impact modéré par défaut
            )
        
        return contenu_réponse
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'IA: {str(e)}")
        return f"Une erreur s'est produite lors de la communication avec l'IA. Détails: {str(e)}"

def traiter_commande_système(commande):
    """Traite une commande système (commençant par /)"""
    parties = commande.lower().split()
    cmd = parties[0] if parties else ""
    
    if cmd in ["aide", "help", "h"]:
        return """
        == Commandes disponibles ==
        /aide - Affiche cette aide
        /statut - Affiche les informations sur votre personnage
        /journal - Consulter votre journal personnel
        /sauvegarder - Sauvegarde la partie actuelle
        /charger - Charge une partie sauvegardée
        /clear - Efface l'écran
        /quitter - Quitte le jeu
        
        == Commandes de jeu ==
        explorer - Explorer les environs
        parler à [personnage] - Dialoguer avec un personnage
        examiner [objet/lieu] - Observer en détail
        utiliser [objet/compétence] - Utiliser un objet ou une compétence
        """
    elif cmd in ["statut", "status", "stat"]:
        if not config_jeu.personnage_actuel:
            return "Aucun personnage actif."
        
        perso = config_jeu.personnage_actuel
        return f"""
        == {perso.nom} ==
        Race: {perso.race}
        Âge: {perso.age} ans
        Origine: {perso.origine} de {perso.lieu_origine}
        Compétence principale: {perso.competence_initiale}
        Santé: {perso.etat_sante}
        
        Apparence: {perso.taille if hasattr(perso, 'taille') else ''}, {perso.corpulence if hasattr(perso, 'corpulence') else ''}
        Cheveux: {perso.couleur_cheveux if hasattr(perso, 'couleur_cheveux') else ''}, {perso.style_cheveux if hasattr(perso, 'style_cheveux') else ''}
        Yeux: {perso.couleur_yeux if hasattr(perso, 'couleur_yeux') else ''}
        Peau: {perso.couleur_peau if hasattr(perso, 'couleur_peau') else ''}
        
        Niveau: {perso.niveau if hasattr(perso, 'niveau') else '1'}
        PV: {perso.pv if hasattr(perso, 'pv') else '100'}/{perso.pv_max if hasattr(perso, 'pv_max') else '100'}
        Mana: {perso.mana if hasattr(perso, 'mana') else '100'}/{perso.mana_max if hasattr(perso, 'mana_max') else '100'}
        
        Force: {perso.force if hasattr(perso, 'force') else '10'}
        Intelligence: {perso.intelligence if hasattr(perso, 'intelligence') else '10'}
        Agilité: {perso.agilité if hasattr(perso, 'agilité') else '10'}
        """
    elif cmd in ["journal"]:
        if not config_jeu.personnage_actuel or not hasattr(config_jeu.personnage_actuel, 'journal'):
            return "Aucune entrée dans votre journal pour le moment."
        
        perso = config_jeu.personnage_actuel
        if not perso.journal:
            return "Votre journal est vide pour le moment. Vos expériences s'y inscriront au fil de vos aventures."
        
        journal = "== Journal personnel ==\n\n"
        for i, entrée in enumerate(perso.journal, 1):
            journal += f"{i}. {entrée['texte']}\n"
        
        return journal
    elif cmd in ["sauvegarder", "save"]:
        # Code pour sauvegarder la partie
        return "Partie sauvegardée avec succès!"
    elif cmd in ["charger", "load"]:
        # Code pour charger une partie
        return "Partie chargée avec succès!"
    elif cmd in ["clear", "cls"]:
        effacer_écran()
        return ""
    elif cmd in ["quitter", "quit", "exit"]:
        config_jeu.état = ÉtatJeu.QUITTER
        return "Au revoir!"
    else:
        return f"Commande '/{cmd}' inconnue. Tapez /aide pour voir la liste des commandes."

def créer_nouveau_personnage():
    """Interface de création de personnage avec choix prédéfinis"""
    imprimer_titre()
    print(f"{Couleur.CYAN}╔════════════ CRÉATION DE PERSONNAGE ════════════╗{Couleur.RESET}")
    
    # Générer un ID unique pour le personnage
    personnage_id = str(uuid.uuid4())
    
    # Nom du personnage (toujours obligatoire)
    nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage (obligatoire): {Couleur.RESET}")
    while not nom:
        imprimer_erreur("Le nom ne peut pas être vide!")
        nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage (obligatoire): {Couleur.RESET}")
    
    # Race (choix ou aléatoire)
    race = afficher_menu_choix("Choisissez votre race", RACES)
    
    # Âge initial (choix ou aléatoire)
    ages_possibles = ["0 (Naissance)", "5 (Enfance)", "10 (Enfance tardive)", "15 (Adolescence)", "20 (Jeune adulte)", "30 (Adulte)", "45 (Âge mûr)"]
    age_choisi = afficher_menu_choix("Choisissez l'âge de départ", ages_possibles)
    
    # Extraction de l'âge numérique du choix
    age = int(age_choisi.split()[0])
    
    # Caractéristiques physiques - Toutes avec choix prédéfinis
    imprimer_système("Définition des caractéristiques physiques")
    
    # Taille (relative à l'âge)
    if age < 15:
        tailles = ["Petit(e) pour son âge", "Normal(e) pour son âge", "Grand(e) pour son âge"]
    else:
        tailles = ["Petit(e)", "De taille moyenne", "Grand(e)", "Très grand(e)"]
    
    taille = afficher_menu_choix("Taille", tailles)
    
    # Corpulence
    corpulences = ["Mince", "Athlétique", "Moyenne", "Robuste", "Enveloppé(e)"]
    corpulence = afficher_menu_choix("Corpulence", corpulences)
    
    # Couleur de cheveux
    couleurs_cheveux = ["Blonds", "Châtains", "Bruns", "Noirs", "Roux", "Gris", "Blancs", "Argentés", "Colorés (bleus, verts, etc.)"]
    couleur_cheveux = afficher_menu_choix("Couleur de cheveux", couleurs_cheveux)
    
    # Style de cheveux
    styles_cheveux = ["Courts", "Mi-longs", "Longs", "Bouclés", "Ondulés", "Raides", "Rasés", "En queue de cheval", "Tressés"]
    style_cheveux = afficher_menu_choix("Style de cheveux", styles_cheveux)
    
    # Couleur des yeux
    couleurs_yeux = ["Bleus", "Verts", "Marrons", "Noirs", "Gris", "Noisette", "Ambre", "Hétérochromie (deux couleurs)"]
    couleur_yeux = afficher_menu_choix("Couleur des yeux", couleurs_yeux)
    
    # Couleur de peau
    couleurs_peau = ["Très pâle", "Claire", "Moyenne", "Bronzée", "Foncée", "Très foncée"]
    couleur_peau = afficher_menu_choix("Couleur de peau", couleurs_peau)
    
    # Trait distinctif (optionnel)
    traits_distinctifs = ["Aucun", "Cicatrice au visage", "Tache de naissance", "Hétérochromie", "Yeux brillants", "Tatouage", "Marque de naissance", "Dents pointues", "Oreilles pointues"]
    trait_distinctif = afficher_menu_choix("Trait distinctif", traits_distinctifs)
    if trait_distinctif == "Aucun":
        trait_distinctif = ""
    
    # Construction de l'apparence complète
    if age == 0:
        description = f"Un nouveau-né avec quelques cheveux {couleur_cheveux.lower()} et des yeux {couleur_yeux.lower()}. Peau {couleur_peau.lower()}."
        if trait_distinctif:
            description += f" {trait_distinctif}."
    else:
        description = f"{taille}, de corpulence {corpulence.lower()}, avec des cheveux {couleur_cheveux.lower()} {style_cheveux.lower()} et des yeux {couleur_yeux.lower()}. Peau {couleur_peau.lower()}."
        if trait_distinctif:
            description += f" Trait distinctif: {trait_distinctif}."
    
    # Origine sociale
    origines = ["Noble", "Bourgeois", "Marchand", "Artisan", "Paysan", "Esclave", "Orphelin", "Érudit", "Aventurier", "Religieux"]
    origine = afficher_menu_choix("Origine sociale", origines)
    
    # Lieu d'origine
    lieux_possibles = ["Royaume de Milis", "Empire de Shirone", "République de Tortus", "Forêt de Biheiril", "Désert d'Asthrea", 
                      "Cité d'Azura", "Montagnes de Faren", "Vallée de Drakon", "Village de pêcheurs", "Cité impériale"]
    lieu_origine = afficher_menu_choix("Lieu d'origine", lieux_possibles)
    
    # Compétence initiale (uniquement si âge >= 5)
    if age >= 5:
        competences = ["Magie élémentaire", "Combat à l'épée", "Archerie", "Guérison", "Alchimie", 
                      "Discrétion", "Érudition", "Survie", "Art oratoire", "Artisanat"]
        competence_initiale = afficher_menu_choix("Compétence initiale", competences)
    else:
        competence_initiale = "Aucune (trop jeune)"
        imprimer_système("Personnage trop jeune pour avoir une compétence initiale.")
    
    # État de santé
    etats_sante = ["Excellente forme", "Forme normale", "Faible constitution", "Maladie chronique légère", "Handicap mineur"]
    
    # Pondération pour nouveau-nés favorisant une meilleure santé
    if age == 0:
        poids = [30, 60, 5, 3, 2]
        etat_sante_index = random.choices(range(len(etats_sante)), weights=poids, k=1)[0]
        etat_sante_options = [etats_sante[etat_sante_index]] + ["Autre choix"]
        etat_sante = afficher_menu_choix(f"État de santé (suggestion: {etats_sante[etat_sante_index]})", etats_sante)
    else:
        etat_sante = afficher_menu_choix("État de santé", etats_sante)
    
    # Secret ou particularité (avec options prédéfinies)
    secrets = ["Aucun", 
              "Descendant d'une lignée noble déchue", 
              "Peut voir des esprits", 
              "Né pendant une rare éclipse", 
              "Marque mystérieuse sur le corps", 
              "Résistance naturelle à la magie", 
              "Affinité spéciale avec les animaux", 
              "Rêves prophétiques occasionnels", 
              "Mémoire eidétique (parfaite)"]
    
    secret = afficher_menu_choix("Secret ou particularité", secrets)
    if secret == "Aucun":
        secret = ""
    
    # Créer la biographie complète adaptée à l'âge
    if age == 0:
        biographie = f"{nom} vient de naître dans une famille {origine.lower()} de {lieu_origine}. "
        if secret:
            biographie += f"Une particularité l'accompagne dès sa naissance: {secret}. "
        biographie += f"Sa santé est {etat_sante.lower()}."
    else:
        biographie = f"Originaire de {lieu_origine}, {nom} est né(e) dans une famille {origine.lower()}. "
        biographie += f"À {age} ans, {nom} "
        if competence_initiale != "Aucune (trop jeune)":
            biographie += f"possède déjà des talents en {competence_initiale.lower()}. "
        else:
            biographie += f"est encore trop jeune pour avoir développé des compétences particulières. "
        if secret:
            biographie += f"Un secret l'accompagne: {secret}. "
        biographie += f"Sa santé est caractérisée par une {etat_sante.lower()}."
    
    # Calculer les attributs de base en fonction de l'âge
    niveau = 1
    pv = 100
    pv_max = 100
    mana = 100
    mana_max = 100
    force = 10
    intelligence = 10
    agilité = 10
        
    # Si c'est un enfant, ajuster les statistiques
    if age < 15:
        pv = 50
        pv_max = 50
        mana = 50
        mana_max = 50
        force = 5
        intelligence = age  # Croît avec l'âge
        agilité = 8
    
    try:
        # Créer le personnage avec l'ID et les paramètres minimaux obligatoires
        personnage = Personnage(
            id=personnage_id,  # ID requis
            nom=nom,
            race=race
        )
        
        # Ajouter tous les autres attributs après création
        personnage.age = age
        personnage.description = description
        personnage.taille = taille
        personnage.corpulence = corpulence
        personnage.couleur_cheveux = couleur_cheveux
        personnage.style_cheveux = style_cheveux
        personnage.couleur_yeux = couleur_yeux
        personnage.couleur_peau = couleur_peau
        personnage.trait_distinctif = trait_distinctif
        personnage.origine = origine
        personnage.lieu_origine = lieu_origine
        personnage.competence_initiale = competence_initiale
        personnage.etat_sante = etat_sante
        personnage.biographie = biographie
        personnage.secret = secret
        
        # Attributs de jeu
        personnage.niveau = niveau
        personnage.pv = pv
        personnage.pv_max = pv_max
        personnage.mana = mana
        personnage.mana_max = mana_max
        personnage.force = force
        personnage.intelligence = intelligence
        personnage.agilité = agilité
        
        # Traits de personnalité
        personnage.traits_personnalite = {
            "courage": 50,
            "empathie": 50,
            "ambition": 50,
            "prudence": 50,
            "sociabilite": 50,
            "moralite": 50,
            "temperament": 50
        }
        personnage.historique_actions = []
        personnage.journal = []
        
        config_jeu.personnage_actuel = personnage
        
        imprimer_système(f"Personnage {nom} créé avec succès!")
        time.sleep(2)
        return personnage
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du personnage: {str(e)}")
        
        # En cas d'erreur, essayer une approche alternative
        try:
            imprimer_système("Tentative de création alternative...")
            
            # Créer une classe alternative qui simule l'interface attendue
            class PersonnageAlternatif:
                def __init__(self):
                    self.id = personnage_id
                    self.nom = nom
                    self.race = race
                    self.age = age
                    self.description = description
                    self.taille = taille
                    self.corpulence = corpulence
                    self.couleur_cheveux = couleur_cheveux
                    self.style_cheveux = style_cheveux
                    self.couleur_yeux = couleur_yeux
                    self.couleur_peau = couleur_peau
                    self.trait_distinctif = trait_distinctif
                    self.origine = origine
                    self.lieu_origine = lieu_origine
                    self.competence_initiale = competence_initiale
                    self.etat_sante = etat_sante
                    self.biographie = biographie
                    self.secret = secret
                    self.niveau = niveau
                    self.pv = pv
                    self.pv_max = pv_max
                    self.mana = mana
                    self.mana_max = mana_max
                    self.force = force
                    self.intelligence = intelligence
                    self.agilité = agilité
                    self.traits_personnalite = {
                        "courage": 50,
                        "empathie": 50,
                        "ambition": 50,
                        "prudence": 50,
                        "sociabilite": 50,
                        "moralite": 50,
                        "temperament": 50
                    }
                    self.historique_actions = []
                    self.journal = []
            
            personnage = PersonnageAlternatif()
            config_jeu.personnage_actuel = personnage
            
            imprimer_système(f"Personnage {nom} créé avec succès (méthode alternative)!")
            time.sleep(2)
            return personnage
            
        except Exception as e2:
            logger.error(f"Échec de la tentative alternative: {str(e2)}")
            imprimer_erreur(f"Échec de la tentative alternative: {str(e2)}")
            time.sleep(2)
            return None

def démarrer_nouvelle_partie():
    """Démarre une nouvelle partie avec adaptation à l'âge de départ"""
    personnage = créer_nouveau_personnage()
    if not personnage:
        return
    
    config_jeu.état = ÉtatJeu.JEU_ACTIF
    config_jeu.historique = []
    
    # Introduction du jeu adaptée à l'âge de départ
    effacer_écran()
    
    if personnage.age == 0:
        imprimer_narration(f"La vie de {personnage.nom} commence tout juste dans le monde de Mushoku Tensei...")
    else:
        imprimer_narration(f"L'aventure de {personnage.nom}, déjà âgé(e) de {personnage.age} ans, se poursuit dans le monde de Mushoku Tensei...")
    
    # Premier prompt pour l'IA adapté à l'âge
    if personnage.age == 0:
        message_intro = f"""
        Je viens de naître sous le nom de {personnage.nom}, un(e) {personnage.race}.
        {personnage.description if hasattr(personnage, 'description') else ''}
        
        Je suis né(e) dans une famille {personnage.origine.lower()} de {personnage.lieu_origine}.
        Ma santé est {personnage.etat_sante.lower()}.
        
        Décris ma naissance, ma famille et mon environnement immédiat.
        """
    elif personnage.age <= 10:
        message_intro = f"""
        Je suis {personnage.nom}, un(e) jeune {personnage.race} de {personnage.age} ans.
        {personnage.description if hasattr(personnage, 'description') else ''}
        
        J'ai grandi dans une famille {personnage.origine.lower()} à {personnage.lieu_origine}.
        Ma santé est {personnage.etat_sante.lower()}.
        
        Où suis-je en ce moment et que se passe-t-il autour de moi?
        """
    else:
        message_intro = f"""
        Je suis {personnage.nom}, un(e) {personnage.race} de {personnage.age} ans.
        {personnage.description if hasattr(personnage, 'description') else ''}
        
        Je viens d'une famille {personnage.origine.lower()} de {personnage.lieu_origine}.
        J'ai des talents en {personnage.competence_initiale.lower() if personnage.competence_initiale != "Aucune (trop jeune)" else "développement"}.
        
        Où suis-je en ce moment et que se passe-t-il autour de moi?
        """
    
    réponse = traiter_commande(message_intro)
    imprimer_narration(réponse)

def charger_partie():
    """Interface de chargement de partie"""
    imprimer_titre()
    print(f"{Couleur.CYAN}╔═══════════════ CHARGER UNE PARTIE ═══════════════╗{Couleur.RESET}")
    
    # Lister les sauvegardes
    sauvegardes = []
    try:
        # Créer le répertoire s'il n'existe pas
        os.makedirs("sauvegardes", exist_ok=True)
        for fichier in os.listdir("sauvegardes"):
            if fichier.endswith(".json"):
                sauvegardes.append(fichier)
    except Exception as e:
        logger.error(f"Impossible de lister les sauvegardes: {str(e)}")
        imprimer_erreur(f"Impossible de lister les sauvegardes: {str(e)}")
    
    if not sauvegardes:
        imprimer_erreur("Aucune sauvegarde trouvée!")
        input("Appuyez sur Entrée pour revenir au menu principal...")
        return
    
    # Afficher les sauvegardes
    print(f"{Couleur.CYAN}Sauvegardes disponibles:{Couleur.RESET}")
    for i, sauvegarde in enumerate(sauvegardes, 1):
        nom_base = sauvegarde.replace(".json", "")
        print(f"{i}. {nom_base}")
    
    print(f"{len(sauvegardes)+1}. Retour au menu principal")
    
    # Sélection
    choix = -1
    while choix < 1 or choix > len(sauvegardes)+1:
        try:
            choix = int(input(f"{Couleur.JAUNE}Votre choix: {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    if choix == len(sauvegardes)+1:
        return
    
    # Charger la sauvegarde
    nom_fichier = sauvegardes[choix-1]
    try:
        # Code pour charger la sauvegarde
        imprimer_système(f"Sauvegarde {nom_fichier} chargée avec succès!")
        config_jeu.état = ÉtatJeu.JEU_ACTIF
        time.sleep(2)
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {str(e)}")
        imprimer_erreur(f"Erreur lors du chargement: {str(e)}")
        input("Appuyez sur Entrée pour continuer...")

def afficher_options():
    """Interface des options du jeu"""
    while True:
        effacer_écran()
        imprimer_titre()
        print(f"{Couleur.CYAN}╔══════════════════ OPTIONS ══════════════════╗{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 1. Modèle IA: {config_jeu.modèle_ia:<25} {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 2. Mode verbose: {'Activé' if config_jeu.mode_verbose else 'Désactivé':<20} {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 3. Mode debug: {'Activé' if config_jeu.mode_debug else 'Désactivé':<22} {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 4. Retour au menu principal                {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}╚══════════════════════════════════════════════╝{Couleur.RESET}")
        
        choix = input(f"{Couleur.JAUNE}Votre choix: {Couleur.RESET}")
        
        if choix == "1":
            modèles = ["mythomax:latest", "mistral:latest", "llama2:latest"]
            print(f"{Couleur.CYAN}Modèles disponibles:{Couleur.RESET}")
            for i, modèle in enumerate(modèles, 1):
                print(f"{i}. {modèle}")
            
            try:
                choix_modèle = int(input(f"{Couleur.JAUNE}Choisissez un modèle (1-{len(modèles)}): {Couleur.RESET}"))
                if 1 <= choix_modèle <= len(modèles):
                    config_jeu.modèle_ia = modèles[choix_modèle-1]
            except ValueError:
                imprimer_erreur("Choix invalide!")
        
        elif choix == "2":
            config_jeu.mode_verbose = not config_jeu.mode_verbose
        
        elif choix == "3":
            config_jeu.mode_debug = not config_jeu.mode_debug
        
        elif choix == "4":
            break
        
        else:
            imprimer_erreur("Option invalide!")
            time.sleep(1)

def afficher_à_propos():
    """Affiche les informations sur le jeu"""
    effacer_écran()
    imprimer_titre()
    print(f"{Couleur.CYAN}╔══════════════════ À PROPOS ══════════════════╗{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}                                              {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}  Mushoku Tensei Roleplay Agent              {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}  Version {VERSION}                              {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}                                              {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}  Basé sur l'univers de Mushoku Tensei        {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}  Utilise le client IA intégré                {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET}                                              {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}╚══════════════════════════════════════════════╝{Couleur.RESET}")
    input(f"{Couleur.JAUNE}Appuyez sur Entrée pour revenir au menu...{Couleur.RESET}")

def boucle_jeu_principale():
    """Boucle principale du jeu lorsqu'une partie est active"""
    while config_jeu.état == ÉtatJeu.JEU_ACTIF:
        try:
            # Interface de commande
            commande = input(f"\n{Couleur.VERT}> {Couleur.RESET}")
            
            # Commande vide
            if not commande:
                continue
                
            # Traiter la commande
            réponse = traiter_commande(commande)
            
            # Si c'est une commande système qui a changé l'état
            if config_jeu.état != ÉtatJeu.JEU_ACTIF:
                break
                
            # Afficher la réponse avec mise en forme
            if réponse:
                imprimer_séparateur()
                imprimer_narration(réponse)
                imprimer_séparateur()
        
        except KeyboardInterrupt:
            imprimer_système("\nVoulez-vous vraiment quitter? (o/n)")
            confirmation = input().lower()
            if confirmation == "o":
                config_jeu.état = ÉtatJeu.QUITTER
                break
        
        except Exception as e:
            logger.error(f"Erreur inattendue: {str(e)}")
            imprimer_erreur(f"Erreur inattendue: {str(e)}")
            if config_jeu.mode_debug:
                import traceback
                traceback.print_exc()

def démarrer_jeu():
    """Fonction principale pour démarrer le jeu"""
    try:
        # Vérification initiale de l'IA
        imprimer_système("Vérification de la connexion à l'IA...")
        client.chat_completion(
            model="mistral:latest",
            messages=[{"role": "system", "content": "Test de connexion"}],
            max_tokens=5
        )
        imprimer_système("Connexion à l'IA établie!")
    except Exception as e:
        logger.error(f"Impossible de se connecter à l'IA: {str(e)}")
        imprimer_erreur(f"Mode de secours activé (sans IA externe)")
        time.sleep(2)
    
    # Boucle principale du menu
    while config_jeu.état != ÉtatJeu.QUITTER:
        if config_jeu.état == ÉtatJeu.MENU_PRINCIPAL:
            afficher_menu_principal()
            choix = input(f"{Couleur.JAUNE}Votre choix: {Couleur.RESET}")
            
            if choix == "1":
                démarrer_nouvelle_partie()
            elif choix == "2":
                charger_partie()
            elif choix == "3":
                afficher_options()
            elif choix == "4":
                afficher_à_propos()
            elif choix == "5":
                config_jeu.état = ÉtatJeu.QUITTER
            else:
                imprimer_erreur("Option invalide!")
                time.sleep(1)
        
        elif config_jeu.état == ÉtatJeu.JEU_ACTIF:
            boucle_jeu_principale()
    
    imprimer_système("Merci d'avoir joué à Mushoku Tensei RP!")
    time.sleep(2)

# Démarrage du jeu lorsque le script est exécuté directement
if __name__ == "__main__":
    démarrer_jeu()


# ===========================================
# === BLOC 14/14 : PERSONNALISATION AVANCÉE ===
# ===========================================

# Structures de données pour la personnalisation avancée des personnages
# Ces systèmes enrichissent l'expérience RPG dans l'univers de Mushoku Tensei

# Origines Familiales et Sociales
ORIGINES = {
    "Noblesse": {
        "Haute noblesse": {"intelligence": +4, "mana_max": +30, "prestige": +10, "richesse": +10},
        "Petite noblesse": {"intelligence": +2, "mana_max": +15, "prestige": +5, "richesse": +5},
        "Famille déchue": {"intelligence": +2, "volonté": +3, "prestige": -2, "richesse": -2}
    },
    "Roturier": {
        "Marchand": {"charisme": +3, "négociation": +5, "richesse": +3},
        "Artisan": {"dextérité": +4, "perception": +3, "création": +5},
        "Fermier": {"endurance": +5, "force": +2, "survie": +3},
        "Aventurier": {"agilité": +3, "perception": +3, "survie": +5}
    },
    "Marginaux": {
        "Orphelin": {"ruse": +4, "discrétion": +3, "survie": +5, "richesse": -5},
        "Criminel": {"discrétion": +5, "intimidation": +3, "ruse": +3, "prestige": -5},
        "Nomade": {"survie": +5, "perception": +3, "langues": +2}
    },
    "Mystique": {
        "Famille de mages": {"mana_max": +50, "intelligence": +3, "connaissance_magique": +5},
        "Prophétisé": {"chance": +5, "destinée": +3, "volonté": +3},
        "Lignée ancienne": {"affinité_élémentaire": +1, "résistance_magique": +3}
    }
}

# Talents Innés et Acquis
TALENTS = [
    # Talents communs (50% chance)
    {"nom": "Apprentissage rapide", "effet": {"vitesse_apprentissage": 1.2}, "rareté": "commun"},
    {"nom": "Force naturelle", "effet": {"force": +3}, "rareté": "commun"},
    {"nom": "Résistance physique", "effet": {"pv_max": +20}, "rareté": "commun"},
    
    # Talents rares (30% chance)
    {"nom": "Affinité magique", "effet": {"coût_mana": 0.8}, "rareté": "rare"},
    {"nom": "Mémoire eidétique", "effet": {"intelligence": +5, "apprentissage_sorts": 1.3}, "rareté": "rare"},
    
    # Talents épiques (15% chance)
    {"nom": "Vision du futur", "effet": {"esquive": +10, "initiative": +5}, "rareté": "épique"},
    {"nom": "Réservoir magique", "effet": {"mana_max": +50}, "rareté": "épique"},
    
    # Talents légendaires (5% chance)
    {"nom": "Sang de dragon", "effet": {"affinité_feu": 1.5, "résistance_magique": +20}, "rareté": "légendaire"},
    {"nom": "Élu des dieux", "effet": {"toutes_stats": +3, "chance": +10}, "rareté": "légendaire"},
    {"nom": "Laplace Factor", "effet": {"affinité_multiple": True, "potentiel_magique": 1.5}, "rareté": "légendaire"}
]

# Formations et Expériences de Vie
FORMATIONS = {
    "Combat": [
        {"nom": "Académie de chevalerie", "prérequis": {"noblesse": True}, "bonus": {"compétence_épée": +5, "tactique": +3}},
        {"nom": "Mercenaire", "prérequis": {"âge_min": 16}, "bonus": {"survie": +4, "compétence_arme": +3}},
        {"nom": "Garde royal", "prérequis": {"force_min": 15}, "bonus": {"défense": +5, "loyauté": +3}}
    ],
    "Magie": [
        {"nom": "Université de magie", "prérequis": {"intelligence_min": 13}, "bonus": {"mana_max": +30, "connaissance_sorts": +5}},
        {"nom": "Apprenti de sorcier", "prérequis": {"mana_min": 40}, "bonus": {"efficacité_sorts": 1.2, "connaissance_sorts": +3}},
        {"nom": "Formation autodidacte", "prérequis": {}, "bonus": {"créativité_magique": +5, "coût_mana": 0.9}}
    ],
    "Artisanat": [
        {"nom": "Forgeron", "prérequis": {"force_min": 12}, "bonus": {"création_armes": +5, "création_armures": +3}},
        {"nom": "Alchimiste", "prérequis": {"intelligence_min": 14}, "bonus": {"création_potions": +5, "connaissance_herbes": +5}},
        {"nom": "Enchanteur", "prérequis": {"mana_min": 60}, "bonus": {"enchantement": +5}}
    ],
    "Autre": [
        {"nom": "Voyageur", "prérequis": {}, "bonus": {"langues": +3, "négociation": +2, "connaissance_monde": +5}},
        {"nom": "Érudit", "prérequis": {"intelligence_min": 15}, "bonus": {"connaissance_histoire": +5, "recherche": +5}},
        {"nom": "Survivaliste", "prérequis": {}, "bonus": {"survie": +5, "pistage": +3, "premiers_soins": +3}}
    ]
}

# Système d'Affinités Élémentaires Avancé
class AffinitéÉlémentaire:
    def __init__(self, personnage):
        self.éléments = {}
        self.calculer_affinités(personnage)
    
    def calculer_affinités(self, personnage):
        # Base aléatoire
        for élément in ELEMENTS:
            valeur_base = random.randint(1, 10)
            
            # Modificateur racial
            if personnage.race == "Elfe" and élément in ["Eau", "Vent"]:
                valeur_base += 3
            elif personnage.race == "Migurd" and élément == "Eau":
                valeur_base += 5
            elif personnage.race == "Supard" and élément == "Feu":
                valeur_base += 5
                
            # Modificateurs génétiques (origine familiale)
            if hasattr(personnage, 'origine_sociale') and personnage.origine_sociale == "Famille de mages":
                valeur_base += 2
                
            # Affinité rare pour magie avancée
            if hasattr(personnage, 'talents') and "Laplace Factor" in personnage.talents and élément in ["Temps", "Espace"]:
                valeur_base += 10
                
            self.éléments[élément] = min(valeur_base, 10)  # Maximum 10
        
    def élément_principal(self):
        return max(self.éléments.items(), key=lambda x: x[1])[0]
        
    def éléments_secondaires(self):
        éléments_triés = sorted(self.éléments.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in éléments_triés[1:3]]  # Les 2 suivants
    
    def __str__(self):
        principal = self.élément_principal()
        secondaires = self.éléments_secondaires()
        return f"Principal: {principal} ({self.éléments[principal]}), Secondaires: {secondaires[0]} ({self.éléments[secondaires[0]]}), {secondaires[1]} ({self.éléments[secondaires[1]]})"

# Classe Personnage Améliorée
@dataclass
class PersonnageAvancé:
    nom: str
    race: str
    description: str
    origine_sociale: str = ""
    origine_catégorie: str = ""
    formation: str = ""
    talents: List[str] = field(default_factory=list)
    niveau: int = 1
    pv: int = 100
    pv_max: int = 100
    mana: int = 50
    mana_max: int = 50
    force: int = 10
    intelligence: int = 10
    agilité: int = 10
    charisme: int = 10
    perception: int = 10
    endurance: int = 10
    volonté: int = 10
    chance: int = 10
    dextérité: int = 10
    compétences: List[Dict] = field(default_factory=list)
    inventaire: List[Dict] = field(default_factory=list)
    or_: int = 100
    prestige: int = 0
    richesse: int = 0
    expérience: int = 0
    expérience_requise: int = 1000
    affinités: Any = None
    équipement: Dict = field(default_factory=lambda: {
        "arme": None,
        "armure": None,
        "casque": None,
        "gants": None,
        "bottes": None,
        "accessoire1": None,
        "accessoire2": None
    })
    
    def __post_init__(self):
        # Ajuster selon la race
        self.ajuster_stats_raciales()
        
        # Ajouter des objets de base à l'inventaire
        self.inventaire.append({"nom": "Potion de soin", "quantité": 3, "effet": {"pv": +30}})
        self.inventaire.append({"nom": "Potion de mana", "quantité": 2, "effet": {"mana": +20}})
        self.inventaire.append({"nom": "Ration", "quantité": 5, "effet": {"faim": -30}})
        
    def ajuster_stats_raciales(self):
        # Implémentation des différences raciales
        if self.race == "Humain":
            self.force += 2
            self.intelligence += 2
            self.charisme += 1
        elif self.race == "Elfe":
            self.intelligence += 4
            self.agilité += 2
            self.mana_max += 20
            self.mana += 20
            self.force -= 1
            self.endurance -= 1
        elif self.race == "Nain":
            self.force += 5
            self.endurance += 3
            self.pv_max += 20
            self.pv += 20
            self.agilité -= 1
        elif self.race == "Beastfolk":
            self.agilité += 5
            self.force += 2
            self.perception += 3
            self.intelligence -= 1
        elif self.race == "Migurd":
            self.intelligence += 5
            self.mana_max += 30
            self.mana += 30
            self.pv_max -= 10
            self.pv -= 10
            self.perception += 2
        elif self.race == "Supard":
            self.force += 8
            self.pv_max += 30
            self.pv += 30
            self.intelligence -= 2
            self.endurance += 4
        elif self.race == "Démon":
            self.force += 3
            self.intelligence += 3
            self.mana_max += 10
            self.mana += 10
            self.charisme += 2

# Fonctions d'application des bonus
def appliquer_bonus_origine(personnage, bonus):
    """Applique les bonus liés à l'origine sociale"""
    for attribut, valeur in bonus.items():
        if hasattr(personnage, attribut):
            setattr(personnage, attribut, getattr(personnage, attribut) + valeur)
        else:
            # Pour les attributs spéciaux non définis dans la classe
            logger.info(f"Attribut spécial '{attribut}' avec valeur {valeur} sera géré séparément")

def appliquer_bonus_talents(personnage, talents):
    """Applique les bonus liés aux talents"""
    for talent in talents:
        for attribut, valeur in talent["effet"].items():
            if attribut == "toutes_stats":
                # Cas spécial: bonus à toutes les statistiques principales
                for stat in ["force", "intelligence", "agilité", "charisme", "perception", "endurance", "volonté", "chance"]:
                    if hasattr(personnage, stat):
                        setattr(personnage, stat, getattr(personnage, stat) + valeur)
            elif hasattr(personnage, attribut):
                # Cas normal: un attribut spécifique
                if isinstance(valeur, (int, float)):
                    setattr(personnage, attribut, getattr(personnage, attribut) + valeur)
                elif isinstance(valeur, (float)) and valeur < 2.0:  # Multiplicateur
                    setattr(personnage, attribut, getattr(personnage, attribut) * valeur)
            else:
                # Attributs spéciaux
                logger.info(f"Attribut spécial '{attribut}' du talent {talent['nom']} sera géré séparément")

def appliquer_bonus_formation(personnage, formation):
    """Applique les bonus liés à la formation"""
    for attribut, valeur in formation["bonus"].items():
        if hasattr(personnage, attribut):
            if isinstance(valeur, (int, float)):
                setattr(personnage, attribut, getattr(personnage, attribut) + valeur)
        else:
            # Pour les compétences spéciales
            if not hasattr(personnage, "bonus_compétences"):
                personnage.bonus_compétences = {}
            personnage.bonus_compétences[attribut] = valeur

# Interface de création de personnage avancée
def créer_nouveau_personnage_avancé():
    """Interface avancée de création de personnage"""
    imprimer_titre()
    print(f"{Couleur.CYAN}╔════════════ CRÉATION DE PERSONNAGE AVANCÉE ════════════╗{Couleur.RESET}")
    
    # Étape 1: Informations de base
    nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage: {Couleur.RESET}")
    while not nom:
        imprimer_erreur("Le nom ne peut pas être vide!")
        nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage: {Couleur.RESET}")
    
    # Étape 2: Sélection de la race
    print(f"{Couleur.CYAN}Choisissez votre race:{Couleur.RESET}")
    for i, race in enumerate(RACES, 1):
        print(f"{i}. {race}")
    
    choix_race = -1
    while choix_race < 1 or choix_race > len(RACES):
        try:
            choix_race = int(input(f"{Couleur.JAUNE}Votre choix (1-{len(RACES)}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    race = RACES[choix_race-1]
    
    # Étape 3: Origine familiale
    origines_catégories = list(ORIGINES.keys())
    print(f"{Couleur.CYAN}Choisissez votre origine sociale:{Couleur.RESET}")
    for i, catégorie in enumerate(origines_catégories, 1):
        print(f"{i}. {catégorie}")
        
    choix_catégorie = -1
    while choix_catégorie < 1 or choix_catégorie > len(origines_catégories):
        try:
            choix_catégorie = int(input(f"{Couleur.JAUNE}Votre choix (1-{len(origines_catégories)}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    catégorie = origines_catégories[choix_catégorie-1]
    origines_spécifiques = list(ORIGINES[catégorie].keys())
    
    print(f"{Couleur.CYAN}Choisissez votre origine spécifique:{Couleur.RESET}")
    for i, origine in enumerate(origines_spécifiques, 1):
        print(f"{i}. {origine}")
        
    choix_origine = -1
    while choix_origine < 1 or choix_origine > len(origines_spécifiques):
        try:
            choix_origine = int(input(f"{Couleur.JAUNE}Votre choix (1-{len(origines_spécifiques)}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    origine = origines_spécifiques[choix_origine-1]
    
    # Étape 4: Détermination des talents (partiellement aléatoire)
    talents = []
    
    # Talent commun garanti
    talents_communs = [t for t in TALENTS if t["rareté"] == "commun"]
    talents.append(random.choice(talents_communs))
    
    # Chance pour un talent rare
    if random.random() < 0.5:  # 50% chance
        talents_rares = [t for t in TALENTS if t["rareté"] == "rare"]
        talents.append(random.choice(talents_rares))
    
    # Petite chance pour un talent épique
    if random.random() < 0.2:  # 20% chance
        talents_épiques = [t for t in TALENTS if t["rareté"] == "épique"]
        talents.append(random.choice(talents_épiques))
    
    # Très faible chance pour un talent légendaire
    if random.random() < 0.05:  # 5% chance
        talents_légendaires = [t for t in TALENTS if t["rareté"] == "légendaire"]
        talents.append(random.choice(talents_légendaires))
    
    print(f"{Couleur.CYAN}Talents déterminés:{Couleur.RESET}")
    for talent in talents:
        print(f"• {talent['nom']} ({talent['rareté'].capitalize()})")
    
    # Étape 5: Historique/Formation
    print(f"{Couleur.CYAN}Choisissez votre formation ou expérience passée:{Couleur.RESET}")
    catégories_formation = list(FORMATIONS.keys())
    
    for i, cat in enumerate(catégories_formation, 1):
        print(f"{i}. {cat}")
        
    choix_cat_form = -1
    while choix_cat_form < 1 or choix_cat_form > len(catégories_formation):
        try:
            choix_cat_form = int(input(f"{Couleur.JAUNE}Votre choix (1-{len(catégories_formation)}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    cat_formation = catégories_formation[choix_cat_form-1]
    formations_disponibles = FORMATIONS[cat_formation]
    
    print(f"{Couleur.CYAN}Choisissez votre formation spécifique:{Couleur.RESET}")
    for i, form in enumerate(formations_disponibles, 1):
        print(f"{i}. {form['nom']}")
    
    choix_formation = -1
    while choix_formation < 1 or choix_formation > len(formations_disponibles):
        try:
            choix_formation = int(input(f"{Couleur.JAUNE}Votre choix (1-{len(formations_disponibles)}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    formation = formations_disponibles[choix_formation-1]
    
    # Étape 6: Description personnalisée
    print(f"{Couleur.CYAN}Décrivez brièvement la personnalité et l'apparence de votre personnage:{Couleur.RESET}")
    description = input(f"{Couleur.JAUNE}> {Couleur.RESET}")
    
    # Créer le personnage avec toutes ces informations
    personnage = PersonnageAvancé(
        nom=nom,
        race=race,
        description=description,
        origine_sociale=origine,
        origine_catégorie=catégorie,
        formation=formation['nom'],
        talents=[t['nom'] for t in talents]
    )
    
    # Appliquer tous les bonus
    appliquer_bonus_origine(personnage, ORIGINES[catégorie][origine])
    appliquer_bonus_talents(personnage, talents)
    appliquer_bonus_formation(personnage, formation)
    
    # Générer les affinités élémentaires
    personnage.affinités = AffinitéÉlémentaire(personnage)
    
    # Finaliser la création
    imprimer_système(f"Personnage {nom} créé avec succès!")
    
    # Afficher un récapitulatif
    print()
    imprimer_système("=== Récapitulatif du Personnage ===")
    imprimer_système(f"Nom: {personnage.nom} ({personnage.race})")
    imprimer_système(f"Origine: {personnage.origine_sociale} ({personnage.origine_catégorie})")
    imprimer_système(f"Formation: {personnage.formation}")
    imprimer_système(f"Talents: {', '.join(personnage.talents)}")
    imprimer_système(f"Affinité élémentaire principale: {personnage.affinités.élément_principal()}")
    
    print()
    imprimer_système("=== Statistiques ===")
    imprimer_système(f"PV: {personnage.pv}/{personnage.pv_max}")
    imprimer_système(f"Mana: {personnage.mana}/{personnage.mana_max}")
    imprimer_système(f"Force: {personnage.force}")
    imprimer_système(f"Intelligence: {personnage.intelligence}")
    imprimer_système(f"Agilité: {personnage.agilité}")
    imprimer_système(f"Charisme: {personnage.charisme}")
    imprimer_système(f"Perception: {personnage.perception}")
    imprimer_système(f"Endurance: {personnage.endurance}")
    imprimer_système(f"Volonté: {personnage.volonté}")
    imprimer_système(f"Chance: {personnage.chance}")
    
    time.sleep(3)
    return personnage

# Remplace la fonction originale de création de personnage par la version avancée
# Pour utiliser la création avancée, remplacez la fonction dans le bloc 13
def intégrer_système_avancé():
    """Remplace le système de création de personnage standard par le système avancé"""
    # Si vous souhaitez utiliser la version avancée, décommentez la ligne ci-dessous
    # et commentez la fonction originale dans le bloc 13
    # globals()["créer_nouveau_personnage"] = créer_nouveau_personnage_avancé
    global Personnage
    Personnage = PersonnageAvancé
    global créer_nouveau_personnage
    créer_nouveau_personnage = créer_nouveau_personnage_avancé
    logger.info("Système de personnalisation avancée activé")

# Automatiquement activer le système avancé au chargement du module
intégrer_système_avancé()

# Fonction pour obtenir une description narrative du personnage
def générer_description_personnage(personnage):
    """Génère une description narrative complète du personnage basée sur tous ses attributs"""
    messages = [
        {"role": "system", "content": """Tu es un narrateur pour un jeu de rôle dans l'univers de Mushoku Tensei.
         Génère une description narrative détaillée et immersive pour ce personnage, qui inclut:
         - Son apparence physique basée sur sa race et ses attributs
         - Sa personnalité et son histoire liée à son origine et sa formation
         - Ses talents spéciaux et leur manifestation
         - Ses affinités élémentaires et comment elles se manifestent
         Utilise un ton captivant et mystérieux qui correspond à l'univers de Mushoku Tensei."""},
        {"role": "user", "content": f"""
        Nom: {personnage.nom}
        Race: {personnage.race}
        Description brève: {personnage.description}
        Origine: {personnage.origine_sociale} ({personnage.origine_catégorie})
        Formation: {personnage.formation}
        Talents: {', '.join(personnage.talents)}
        Affinité élémentaire principale: {personnage.affinités.élément_principal()}
        Affinités secondaires: {', '.join(personnage.affinités.éléments_secondaires())}
        
        Statistiques principales:
        Force: {personnage.force}
        Intelligence: {personnage.intelligence}
        Agilité: {personnage.agilité}
        Charisme: {personnage.charisme}
        Perception: {personnage.perception}
        Endurance: {personnage.endurance}
        Volonté: {personnage.volonté}
        Chance: {personnage.chance}
        """}
    ]
    
    try:
        réponse = client.chat_completion(
            model=config_jeu.modèle_ia,
            messages=messages
        )
        
        contenu_réponse = ""
        if isinstance(réponse, dict):
            # Adapter selon la structure exacte retournée
            if "response" in réponse and isinstance(réponse["response"], dict):
                contenu_réponse = réponse["response"].get("content", "")
            elif "response" in réponse and isinstance(réponse["response"], str):
                contenu_réponse = réponse["response"]
            elif "choices" in réponse:
                contenu_réponse = réponse["choices"][0]["message"]["content"]
            else:
                contenu_réponse = str(réponse)
        else:
            contenu_réponse = str(réponse)
        
        return contenu_réponse
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la description du personnage: {str(e)}")
        return f"Un mystérieux aventurier dont l'histoire reste à découvrir..."

# Fonction pour afficher la fiche de personnage détaillée
def afficher_fiche_personnage(personnage):
    """Affiche une fiche complète du personnage avec tous les détails"""
    effacer_écran()
    imprimer_titre()
    print(f"{Couleur.CYAN}╔════════════ FICHE DE PERSONNAGE ════════════╗{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} {personnage.nom:<42} {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}╟──────────────────────────────────────────────╢{Couleur.RESET}")
    
    print(f"{Couleur.CYAN}║{Couleur.RESET} {Couleur.JAUNE}Race:{Couleur.RESET} {personnage.race:<36} {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} {Couleur.JAUNE}Origine:{Couleur.RESET} {personnage.origine_sociale:<33} {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} {Couleur.JAUNE}Formation:{Couleur.RESET} {personnage.formation:<32} {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} {Couleur.JAUNE}Niveau:{Couleur.RESET} {personnage.niveau:<35} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Statistiques principales
    print(f"{Couleur.CYAN}╟─────────────── STATISTIQUES ───────────────────╢{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} PV: {personnage.pv}/{personnage.pv_max:<36} {Couleur.CYAN}║{Couleur.RESET}")
    print(f"{Couleur.CYAN}║{Couleur.RESET} Mana: {personnage.mana}/{personnage.mana_max:<34} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Affichage sur deux colonnes pour les attributs
    stats = [
        ("Force", personnage.force), 
        ("Intelligence", personnage.intelligence),
        ("Agilité", personnage.agilité),
        ("Charisme", personnage.charisme),
        ("Perception", personnage.perception),
        ("Endurance", personnage.endurance),
        ("Volonté", personnage.volonté),
        ("Chance", personnage.chance)
    ]
    
    for i in range(0, len(stats), 2):
        if i+1 < len(stats):
            print(f"{Couleur.CYAN}║{Couleur.RESET} {stats[i][0]}: {stats[i][1]:<10} | {stats[i+1][0]}: {stats[i+1][1]:<10} {Couleur.CYAN}║{Couleur.RESET}")
        else:
            print(f"{Couleur.CYAN}║{Couleur.RESET} {stats[i][0]}: {stats[i][1]:<32} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Talents
    print(f"{Couleur.CYAN}╟────────────────── TALENTS ────────────────────╢{Couleur.RESET}")
    for talent in personnage.talents:
        print(f"{Couleur.CYAN}║{Couleur.RESET} • {talent:<39} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Affinités élémentaires
    print(f"{Couleur.CYAN}╟─────────── AFFINITÉS ÉLÉMENTAIRES ────────────╢{Couleur.RESET}")
    élément_principal = personnage.affinités.élément_principal()
    valeur_principale = personnage.affinités.éléments[élément_principal]
    print(f"{Couleur.CYAN}║{Couleur.RESET} Principal: {élément_principal} ({valeur_principale}/10){' ' * (28 - len(élément_principal))} {Couleur.CYAN}║{Couleur.RESET}")
    
    secondaires = personnage.affinités.éléments_secondaires()
    for élément in secondaires:
        valeur = personnage.affinités.éléments[élément]
        print(f"{Couleur.CYAN}║{Couleur.RESET} Secondaire: {élément} ({valeur}/10){' ' * (27 - len(élément))} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Compétences
    if personnage.compétences:
        print(f"{Couleur.CYAN}╟───────────────── COMPÉTENCES ─────────────────╢{Couleur.RESET}")
        for comp in personnage.compétences:
            if isinstance(comp, dict) and "nom" in comp:
                print(f"{Couleur.CYAN}║{Couleur.RESET} • {comp['nom']:<39} {Couleur.CYAN}║{Couleur.RESET}")
            else:
                print(f"{Couleur.CYAN}║{Couleur.RESET} • {comp:<39} {Couleur.CYAN}║{Couleur.RESET}")
    
    # Inventaire
    if personnage.inventaire:
        print(f"{Couleur.CYAN}╟────────────────── INVENTAIRE ──────────────────╢{Couleur.RESET}")
        for item in personnage.inventaire:
            if isinstance(item, dict) and "nom" in item and "quantité" in item:
                print(f"{Couleur.CYAN}║{Couleur.RESET} • {item['nom']} x{item['quantité']:<34} {Couleur.CYAN}║{Couleur.RESET}")
            else:
                print(f"{Couleur.CYAN}║{Couleur.RESET} • {item:<39} {Couleur.CYAN}║{Couleur.RESET}")
        
        print(f"{Couleur.CYAN}║{Couleur.RESET} Or: {personnage.or_:<38} {Couleur.CYAN}║{Couleur.RESET}")
    
    print(f"{Couleur.CYAN}╚══════════════════════════════════════════════╝{Couleur.RESET}")
    
    input(f"\n{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")

# Remplacer la commande /statut originale par la version avancée
def traiter_commande_système_avancé(commande):
    """Version améliorée de la fonction de traitement des commandes système"""
    parties = commande.lower().split()
    cmd = parties[0] if parties else ""
    
    if cmd in ["aide", "help", "h"]:
        return """
        == Commandes disponibles ==
        /aide - Affiche cette aide
        /statut - Affiche les informations sur votre personnage
        /fiche - Affiche la fiche détaillée de personnage
        /sauvegarder - Sauvegarde la partie actuelle
        /charger - Charge une partie sauvegardée
        /description - Génère une description narrative de votre personnage
        /clear - Efface l'écran
        /quitter - Quitte le jeu
        
        == Commandes de jeu ==
        explorer - Explorer les environs
        parler à [personnage] - Dialoguer avec un personnage
        examiner [objet/lieu] - Observer en détail
        utiliser [objet/compétence] - Utiliser un objet ou une compétence
        """
    elif cmd in ["statut", "status", "stat"]:
        if not config_jeu.personnage_actuel:
            return "Aucun personnage actif."
        afficher_fiche_personnage(config_jeu.personnage_actuel)
        return ""
    elif cmd in ["fiche", "personnage", "sheet"]:
        if not config_jeu.personnage_actuel:
            return "Aucun personnage actif."
        afficher_fiche_personnage(config_jeu.personnage_actuel)
        return ""
    elif cmd in ["description", "desc"]:
        if not config_jeu.personnage_actuel:
            return "Aucun personnage actif."
        description = générer_description_personnage(config_jeu.personnage_actuel)
        return description
    elif cmd in ["sauvegarder", "save"]:
        # Code pour sauvegarder la partie
        return "Partie sauvegardée avec succès!"
    elif cmd in ["charger", "load"]:
        # Code pour charger une partie
        return "Partie chargée avec succès!"
    elif cmd in ["clear", "cls"]:
        effacer_écran()
        return ""
    elif cmd in ["quitter", "quit", "exit"]:
        config_jeu.état = ÉtatJeu.QUITTER
        return "Au revoir!"
    else:
        return f"Commande '/{cmd}' inconnue. Tapez /aide pour voir la liste des commandes."

# Remplacer la fonction du bloc 13
def remplacer_fonctions_originales():
    """Remplace certaines fonctions du bloc 13 par leurs versions améliorées"""
    global traiter_commande_système
    original_traiter_commande_système = traiter_commande_système
    traiter_commande_système = traiter_commande_système_avancé

# Activer le remplacement des fonctions si le script est exécuté directement
if __name__ == "__main__":
    try:
        remplacer_fonctions_originales()
        logger.info("Système de personnalisation avancée intégré avec succès")
        print(f"{Couleur.VERT}Système de personnalisation avancée activé!{Couleur.RESET}")
    except Exception as e:
        logger.error(f"Erreur lors de l'intégration du système avancé: {str(e)}")
        print(f"{Couleur.ROUGE}Erreur lors de l'activation du système avancé: {str(e)}{Couleur.RESET}")

# ===========================================
# === BLOC 15/15 : AVENTURE À CHOIX MULTIPLES ====
# ===========================================

import os
import sys
import time
import random
import re
import json
import logging
from enum import Enum
import uuid
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("aventure_choix.log", encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger("aventure_choix")

# Version du jeu
VERSION = "1.0"

# Modèle IA par défaut (peut être modifié dans les options)
DEFAULT_MODEL = "mythomax:latest"

# Liste des races disponibles
RACES = ["Humain", "Elfe", "Nain", "Demi-Elfe", "Demi-Démon", "Bête-Humain", "Démon"]

#===================================
# CLASSES ET FONCTIONS DU SYSTÈME
#===================================

class Couleur:
    """Codes ANSI pour colorer le texte dans le terminal"""
    RESET = '\033[0m'
    GRAS = '\033[1m'
    SOULIGNÉ = '\033[4m'
    NOIR = '\033[30m'
    ROUGE = '\033[31m'
    VERT = '\033[32m'
    JAUNE = '\033[33m'
    BLEU = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BLANC = '\033[37m'
    FOND_NOIR = '\033[40m'
    FOND_ROUGE = '\033[41m'
    FOND_VERT = '\033[42m'
    FOND_JAUNE = '\033[43m'
    FOND_BLEU = '\033[44m'
    FOND_MAGENTA = '\033[45m'
    FOND_CYAN = '\033[46m'
    FOND_BLANC = '\033[47m'

class ConfigJeu:
    """Configuration globale du jeu"""
    def __init__(self):
        self.modèle_ia = DEFAULT_MODEL
        self.personnage_actuel = None
        self.historique = []
        self.dernière_sauvegarde = None
        self.mode_verbose = False
        self.mode_debug = False
        
# Objet de configuration utilisable dans tout le code
config_jeu = ConfigJeu()

def effacer_écran():
    """Efface l'écran du terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimer_titre():
    """Affiche le titre du jeu"""
    effacer_écran()
    print(f"{Couleur.CYAN}{Couleur.GRAS}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                                                           ║")
    print("║              MUSHOKU TENSEI ROLEPLAY AGENT                ║")
    print(f"║            AVENTURE IMMERSIVE À CHOIX v{VERSION}               ║")
    print("║                                                           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Couleur.RESET}")

def imprimer_séparateur():
    """Affiche un séparateur dans le terminal"""
    print(f"{Couleur.CYAN}{'─' * 60}{Couleur.RESET}")

def imprimer_message(texte, couleur=Couleur.BLANC, délai=0.03):
    """Affiche un texte caractère par caractère pour un effet d'écriture"""
    for caractère in texte:
        print(f"{couleur}{caractère}{Couleur.RESET}", end='', flush=True)
        time.sleep(délai)
    print()

def imprimer_narration(texte):
    """Affiche un texte de narration"""
    imprimer_message(texte, Couleur.JAUNE, 0.02)

def imprimer_dialogue(personnage, texte):
    """Affiche un dialogue avec le nom du personnage"""
    print(f"{Couleur.CYAN}{Couleur.GRAS}{personnage}{Couleur.RESET}: ", end='')
    imprimer_message(texte, Couleur.BLANC)

def imprimer_action(texte):
    """Affiche une action réalisée par le joueur"""
    print(f"{Couleur.VERT}» {texte}{Couleur.RESET}")

def imprimer_système(texte):
    """Affiche un message système"""
    print(f"{Couleur.MAGENTA}[Système] {texte}{Couleur.RESET}")

def imprimer_erreur(texte):
    """Affiche un message d'erreur"""
    print(f"{Couleur.ROUGE}[ERREUR] {texte}{Couleur.RESET}")

def afficher_menu_choix(titre, options, permettre_aleatoire=True):
    """Affiche un menu de choix et retourne l'option sélectionnée"""
    print(f"{Couleur.CYAN}{titre}:{Couleur.RESET}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    if permettre_aleatoire:
        print(f"0. Aléatoire")
    
    while True:
        choix = input(f"{Couleur.JAUNE}Votre choix {f'(0-{len(options)})' if permettre_aleatoire else f'(1-{len(options)})' }: {Couleur.RESET}")
        
        # Option aléatoire
        if choix == "0" and permettre_aleatoire:
            choix_index = random.randint(0, len(options) - 1)
            imprimer_système(f"Choix aléatoire : {options[choix_index]}")
            return options[choix_index]
        
        # Option spécifique
        try:
            choix_num = int(choix)
            if 1 <= choix_num <= len(options):
                return options[choix_num - 1]
            else:
                imprimer_erreur(f"Veuillez entrer un nombre entre {'0' if permettre_aleatoire else '1'} et {len(options)}.")
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")

# Client IA pour interagir avec l'IA (version simplifiée qui ne nécessite pas d'API externe)
class ClientIA:
    """Gère l'interaction avec l'IA pour générer des réponses"""
    
    def __init__(self):
        self.debug_mode = False
        self.last_prompt = ""
    
    def chat_completion(self, model, messages, max_tokens=None):
        """
        Simule un appel à une API d'IA. Dans une version réelle,
        cette méthode appelerait une API comme OpenAI, Anthropic ou Ollama.
        """
        try:
            # Enregistrer le dernier prompt pour débogage
            self.last_prompt = messages[-1]["content"] if messages else ""
            
            if self.debug_mode:
                logger.debug(f"Appel à {model} avec prompt: {self.last_prompt[:100]}...")
            
            # Contexte du dernier message
            context = messages[-1]["content"] if messages else ""
            
            # Générer une réponse appropriée basée sur des mots-clés dans le contexte
            response = self._generate_contextual_response(context)
            
            return {"response": response}
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'IA: {str(e)}")
            return {"response": "Je dois réfléchir à la suite de l'histoire..."}

    def _generate_contextual_response(self, context):
        """Génère une réponse basée sur le contexte de l'entrée"""
        # Extraire les mots-clés pour adapter la réponse
        context_lower = context.lower()
        
        # Réponse pour un début de naissance
        if any(mot in context_lower for mot in ["naissance", "nouveau-né", "viens de naître"]):
            return self._birth_scenario()
        
        # Réponse pour divers scénarios d'exploration
        elif any(mot in context_lower for mot in ["explorer", "environnement", "découvrir"]):
            return self._exploration_scenario()
        
        # Réponse pour des situations de combat
        elif any(mot in context_lower for mot in ["combat", "bataille", "affronter", "attaquer"]):
            return self._combat_scenario()
        
        # Réponse pour des rencontres sociales
        elif any(mot in context_lower for mot in ["parler", "discuter", "négocier", "interroger"]):
            return self._social_scenario()
        
        # Réponse par défaut
        else:
            return self._default_scenario()
    
    def _birth_scenario(self):
        """Scénario de naissance"""
        return """
        Les premiers instants de ta vie sont flous, baignés dans une lumière douce et tamisée. Des voix s'entremêlent autour de toi, certaines familières bien que tu ne puisses pas encore associer des visages à ces sons. Une chaleur réconfortante t'enveloppe alors que tu es blotti dans les bras de ta mère.

        Ton regard encore incertain distingue des silhouettes qui se penchent pour t'observer. Des expressions d'émerveillement, de joie et de soulagement se dessinent sur ces visages.

        "Un enfant magnifique," murmure une voix masculine grave avec une pointe de fierté. "Un don des dieux pour notre famille."

        Que ressens-tu en ces premiers instants de vie ?

        [Choix A] Tu pleures, ressentant la confusion d'être plongé dans ce monde nouveau et inconnu
        [Choix B] Tu restes paisible, observant avec curiosité ces visages qui t'entourent
        [Choix C] Tu tends instinctivement ta petite main vers le visage de ta mère
        [Choix D] Tu fermes les yeux, épuisé par l'expérience intense de la naissance
        """
    
    def _exploration_scenario(self):
        """Scénario d'exploration"""
        return """
        Les alentours sont riches en détails et en possibilités. Des arbres majestueux se dressent de part et d'autre du chemin, leurs feuillages denses filtrant les rayons du soleil qui créent des motifs dansants sur le sol. L'air est empli des chants d'oiseaux et du bruissement des feuilles bercées par la brise.

        Au loin, tu aperçois la silhouette d'une petite structure qui pourrait être une cabane ou un avant-poste. À ta droite, un sentier plus étroit semble s'enfoncer plus profondément dans les bois. De l'autre côté, tu remarques une légère colonne de fumée s'élevant au-dessus des arbres, suggérant peut-être un village ou un campement.

        Que décides-tu de faire à présent ?

        [Choix A] Se diriger vers la structure visible au loin pour l'examiner de plus près
        [Choix B] Emprunter le sentier étroit qui s'enfonce dans les bois
        [Choix C] Investiguer la source de la fumée visible au-dessus des arbres
        [Choix D] Rester sur place et observer plus attentivement les environs immédiats
        """
    
    def _combat_scenario(self):
        """Scénario de combat"""
        return """
        La tension est palpable alors que tu te retrouves face à l'adversité. Ton cœur bat plus vite, l'adrénaline inonde tes veines, aiguisant tes sens. Tu évalues rapidement la situation : la posture de ton adversaire, les armes ou capacités dont il dispose, ainsi que l'environnement qui pourrait jouer en ta faveur ou contre toi.

        Le temps semble ralentir alors que tu envisages tes options. Tu sens le poids de ta propre arme dans ta main, ou l'énergie magique qui circule en toi, prête à être libérée.

        Quelle approche choisis-tu pour ce face-à-face ?

        [Choix A] Attaquer directement et rapidement pour prendre l'avantage de la surprise
        [Choix B] Adopter une posture défensive pour évaluer les capacités de l'adversaire
        [Choix C] Tenter de désamorcer la situation par la parole avant qu'elle ne dégénère
        [Choix D] Chercher un avantage tactique en utilisant l'environnement alentour
        """
    
    def _social_scenario(self):
        """Scénario social"""
        return """
        La personne en face de toi te regarde avec une expression que tu tentes d'interpréter. Ses yeux reflètent un mélange de curiosité et de prudence, tandis que sa posture corporelle suggère une ouverture au dialogue, bien que teintée d'une certaine réserve.

        L'ambiance autour de vous influence cette interaction : les personnes à proximité semblent vaquer à leurs occupations, mais tu sens que certaines pourraient être attentives à votre conversation.

        Comment souhaites-tu aborder cette interaction sociale ?

        [Choix A] Engager une conversation amicale et légère pour établir un rapport de confiance
        [Choix B] Aller droit au but et exprimer clairement ce que tu souhaites obtenir
        [Choix C] Poser des questions pour en apprendre davantage sur cette personne
        [Choix D] Attendre que ton interlocuteur prenne l'initiative de la conversation
        """
    
    def _default_scenario(self):
        """Scénario par défaut"""
        return """
        Le monde de Mushoku Tensei s'étend devant toi, vaste et rempli d'opportunités. Chaque choix pourrait te mener vers de nouvelles aventures, rencontres et découvertes. Tu sens que ton destin est entre tes mains, façonné par les décisions que tu prendras.

        Le moment présent est comme un carrefour où plusieurs chemins s'offrent à toi, chacun porteur de ses propres défis et récompenses.

        Que souhaites-tu faire maintenant ?

        [Choix A] Suivre ton instinct et avancer vers l'inconnu
        [Choix B] Réfléchir à ton passé et aux événements qui t'ont mené jusqu'ici
        [Choix C] Chercher quelqu'un qui pourrait te guider ou t'informer
        [Choix D] Prendre le temps d'observer ton environnement plus attentivement
        """

# Initialisation du client IA
client = ClientIA()

# Classe pour la gestion du personnage
class Personnage:
    """Représente le personnage du joueur avec toutes ses caractéristiques"""
    def __init__(self, id, nom, race):
        # Identificateur unique
        self.id = id
        self.nom = nom
        self.race = race
        
        # Attributs de base (à définir après l'initialisation)
        self.age = 0
        self.description = ""
        self.biographie = ""
        
        # Apparence physique
        self.taille = ""
        self.corpulence = ""
        self.couleur_cheveux = ""
        self.style_cheveux = ""
        self.couleur_yeux = ""
        self.couleur_peau = ""
        self.trait_distinctif = ""
        
        # Origines et compétences
        self.origine = ""
        self.lieu_origine = ""
        self.competence_initiale = ""
        self.etat_sante = ""
        self.secret = ""
        
        # Statistiques
        self.niveau = 1
        self.pv = 100
        self.pv_max = 100
        self.mana = 100
        self.mana_max = 100
        self.force = 10
        self.intelligence = 10
        self.agilité = 10
        
        # Traits de personnalité (0-100)
        self.traits_personnalite = {
            "courage": 50,
            "empathie": 50,
            "ambition": 50,
            "prudence": 50,
            "sociabilite": 50,
            "moralite": 50,
            "temperament": 50
        }
        
        # Historique et journal
        self.historique_actions = []
        self.journal = []

# Fonction de création de personnage
def créer_nouveau_personnage():
    """Interface complète de création de personnage avec choix prédéfinis"""
    imprimer_titre()
    print(f"{Couleur.CYAN}╔════════════ CRÉATION DE PERSONNAGE ════════════╗{Couleur.RESET}")
    
    # Générer un ID unique pour le personnage
    personnage_id = str(uuid.uuid4())
    
    # Nom du personnage (toujours obligatoire)
    nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage (obligatoire): {Couleur.RESET}")
    while not nom:
        imprimer_erreur("Le nom ne peut pas être vide!")
        nom = input(f"{Couleur.JAUNE}Entrez le nom de votre personnage (obligatoire): {Couleur.RESET}")
    
    # Race (choix ou aléatoire)
    race = afficher_menu_choix("Choisissez votre race", RACES)
    
    # Âge initial (choix ou aléatoire)
    ages_possibles = ["0 (Naissance)", "5 (Enfance)", "10 (Enfance tardive)", "15 (Adolescence)", "20 (Jeune adulte)", "30 (Adulte)", "45 (Âge mûr)"]
    age_choisi = afficher_menu_choix("Choisissez l'âge de départ", ages_possibles)
    
    # Extraction de l'âge numérique du choix
    age = int(age_choisi.split()[0])
    
    # Caractéristiques physiques - Toutes avec choix prédéfinis
    imprimer_système("Définition des caractéristiques physiques")
    
    # Taille (relative à l'âge)
    if age < 15:
        tailles = ["Petit(e) pour son âge", "Normal(e) pour son âge", "Grand(e) pour son âge"]
    else:
        tailles = ["Petit(e)", "De taille moyenne", "Grand(e)", "Très grand(e)"]
    
    taille = afficher_menu_choix("Taille", tailles)
    
    # Corpulence
    corpulences = ["Mince", "Athlétique", "Moyenne", "Robuste", "Enveloppé(e)"]
    corpulence = afficher_menu_choix("Corpulence", corpulences)
    
    # Couleur de cheveux
    couleurs_cheveux = ["Blonds", "Châtains", "Bruns", "Noirs", "Roux", "Gris", "Blancs", "Argentés", "Colorés (bleus, verts, etc.)"]
    couleur_cheveux = afficher_menu_choix("Couleur de cheveux", couleurs_cheveux)
    
    # Style de cheveux
    styles_cheveux = ["Courts", "Mi-longs", "Longs", "Bouclés", "Ondulés", "Raides", "Rasés", "En queue de cheval", "Tressés"]
    style_cheveux = afficher_menu_choix("Style de cheveux", styles_cheveux)
    
    # Couleur des yeux
    couleurs_yeux = ["Bleus", "Verts", "Marrons", "Noirs", "Gris", "Noisette", "Ambre", "Hétérochromie (deux couleurs)"]
    couleur_yeux = afficher_menu_choix("Couleur des yeux", couleurs_yeux)
    
    # Couleur de peau
    couleurs_peau = ["Très pâle", "Claire", "Moyenne", "Bronzée", "Foncée", "Très foncée"]
    couleur_peau = afficher_menu_choix("Couleur de peau", couleurs_peau)
    
    # Trait distinctif (optionnel)
    traits_distinctifs = ["Aucun", "Cicatrice au visage", "Tache de naissance", "Hétérochromie", "Yeux brillants", "Tatouage", "Marque de naissance", "Dents pointues", "Oreilles pointues"]
    trait_distinctif = afficher_menu_choix("Trait distinctif", traits_distinctifs)
    if trait_distinctif == "Aucun":
        trait_distinctif = ""
    
    # Construction de l'apparence complète
    if age == 0:
        description = f"Un nouveau-né avec quelques cheveux {couleur_cheveux.lower()} et des yeux {couleur_yeux.lower()}. Peau {couleur_peau.lower()}."
        if trait_distinctif:
            description += f" {trait_distinctif}."
    else:
        description = f"{taille}, de corpulence {corpulence.lower()}, avec des cheveux {couleur_cheveux.lower()} {style_cheveux.lower()} et des yeux {couleur_yeux.lower()}. Peau {couleur_peau.lower()}."
        if trait_distinctif:
            description += f" Trait distinctif: {trait_distinctif}."
    
    # Origine sociale
    origines = ["Noble", "Bourgeois", "Marchand", "Artisan", "Paysan", "Esclave", "Orphelin", "Érudit", "Aventurier", "Religieux"]
    origine = afficher_menu_choix("Origine sociale", origines)
    
    # Lieu d'origine
    lieux_possibles = ["Royaume de Milis", "Empire de Shirone", "République de Tortus", "Forêt de Biheiril", "Désert d'Asthrea", 
                      "Cité d'Azura", "Montagnes de Faren", "Vallée de Drakon", "Village de pêcheurs", "Cité impériale"]
    lieu_origine = afficher_menu_choix("Lieu d'origine", lieux_possibles)
    
    # Compétence initiale (uniquement si âge >= 5)
    if age >= 5:
        competences = ["Magie élémentaire", "Combat à l'épée", "Archerie", "Guérison", "Alchimie", 
                      "Discrétion", "Érudition", "Survie", "Art oratoire", "Artisanat"]
        competence_initiale = afficher_menu_choix("Compétence initiale", competences)
    else:
        competence_initiale = "Aucune (trop jeune)"
        imprimer_système("Personnage trop jeune pour avoir une compétence initiale.")
    
    # État de santé
    etats_sante = ["Excellente forme", "Forme normale", "Faible constitution", "Maladie chronique légère", "Handicap mineur"]
    
    # Pondération pour nouveau-nés favorisant une meilleure santé
    if age == 0:
        poids = [30, 60, 5, 3, 2]
        etat_sante_index = random.choices(range(len(etats_sante)), weights=poids, k=1)[0]
        etat_sante_options = [etats_sante[etat_sante_index]] + ["Autre choix"]
        etat_sante = afficher_menu_choix(f"État de santé (suggestion: {etats_sante[etat_sante_index]})", etats_sante)
    else:
        etat_sante = afficher_menu_choix("État de santé", etats_sante)
    
    # Secret ou particularité (avec options prédéfinies)
    secrets = ["Aucun", 
              "Descendant d'une lignée noble déchue", 
              "Peut voir des esprits", 
              "Né pendant une rare éclipse", 
              "Marque mystérieuse sur le corps", 
              "Résistance naturelle à la magie", 
              "Affinité spéciale avec les animaux", 
              "Rêves prophétiques occasionnels", 
              "Mémoire eidétique (parfaite)"]
    
    secret = afficher_menu_choix("Secret ou particularité", secrets)
    if secret == "Aucun":
        secret = ""
    
    # Créer la biographie complète adaptée à l'âge
    if age == 0:
        biographie = f"{nom} vient de naître dans une famille {origine.lower()} de {lieu_origine}. "
        if secret:
            biographie += f"Une particularité l'accompagne dès sa naissance: {secret}. "
        biographie += f"Sa santé est {etat_sante.lower()}."
    else:
        biographie = f"Originaire de {lieu_origine}, {nom} est né(e) dans une famille {origine.lower()}. "
        biographie += f"À {age} ans, {nom} "
        if competence_initiale != "Aucune (trop jeune)":
            biographie += f"possède déjà des talents en {competence_initiale.lower()}. "
        else:
            biographie += f"est encore trop jeune pour avoir développé des compétences particulières. "
        if secret:
            biographie += f"Un secret l'accompagne: {secret}. "
        biographie += f"Sa santé est caractérisée par une {etat_sante.lower()}."
    
    # Calculer les attributs de base en fonction de l'âge
    niveau = 1
    pv = 100
    pv_max = 100
    mana = 100
    mana_max = 100
    force = 10
    intelligence = 10
    agilité = 10
        
    # Si c'est un enfant, ajuster les statistiques
    if age < 15:
        pv = 50
        pv_max = 50
        mana = 50
        mana_max = 50
        force = 5
        intelligence = age  # Croît avec l'âge
        agilité = 8
    
    try:
        # Créer le personnage avec l'ID et les paramètres minimaux obligatoires
        personnage = Personnage(
            id=personnage_id,  # ID requis
            nom=nom,
            race=race
        )
        
        # Ajouter tous les autres attributs après création
        personnage.age = age
        personnage.description = description
        personnage.taille = taille
        personnage.corpulence = corpulence
        personnage.couleur_cheveux = couleur_cheveux
        personnage.style_cheveux = style_cheveux
        personnage.couleur_yeux = couleur_yeux
        personnage.couleur_peau = couleur_peau
        personnage.trait_distinctif = trait_distinctif
        personnage.origine = origine
        personnage.lieu_origine = lieu_origine
        personnage.competence_initiale = competence_initiale
        personnage.etat_sante = etat_sante
        personnage.biographie = biographie
        personnage.secret = secret
        
        # Attributs de jeu
        personnage.niveau = niveau
        personnage.pv = pv
        personnage.pv_max = pv_max
        personnage.mana = mana
        personnage.mana_max = mana_max
        personnage.force = force
        personnage.intelligence = intelligence
        personnage.agilité = agilité
        
        # Traits de personnalité initialisés plus tôt dans __init__
        
        # Assigner au joueur actuel dans la config globale
        config_jeu.personnage_actuel = personnage
        
        imprimer_système(f"Personnage {nom} créé avec succès!")
        time.sleep(2)
        return personnage
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du personnage: {str(e)}")
        imprimer_erreur(f"Erreur lors de la création du personnage: {str(e)}")
        time.sleep(2)
        return None


#===================================
# CLASSES POUR L'AVENTURE À CHOIX
#===================================

class TypeÉvénement(Enum):
    """Les différents types d'événements possibles dans l'histoire"""
    NORMAL = 0          # Événement narratif standard avec 4 choix
    DÉCISION_MAJEURE = 1 # Décision qui affecte significativement l'histoire
    RENCONTRE = 2       # Rencontre avec un personnage important
    COMBAT = 3          # Séquence de combat
    RÉVÉLATION = 4      # Révélation d'information importante
    ACQUISITION = 5     # Acquisition d'un objet ou compétence
    CONCLUSION = 6      # Fin d'un arc narratif

class Événement:
    """Représente un événement narratif avec des choix associés"""
    
    def __init__(self, id_événement, type_événement, texte, choix, conséquences=None):
        self.id = id_événement
        self.type = type_événement
        self.texte = texte
        self.choix = choix  # liste de str
        self.conséquences = conséquences or {}  # dict: choix_index -> id_événement_suivant
        self.timestamp = datetime.now().isoformat()

class JournalAventure:
    """Enregistre les événements importants de l'aventure"""
    
    def __init__(self):
        self.entrées = []
    
    def ajouter_entrée(self, titre, description, importance=1):
        """
        Ajoute une entrée au journal
        
        Args:
            titre (str): Titre court de l'entrée
            description (str): Description détaillée
            importance (int): Niveau d'importance de 1 (normal) à 3 (majeur)
        """
        self.entrées.append({
            "titre": titre,
            "description": description,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        })
    
    def afficher(self, nb_entrées=None):
        """Affiche les entrées du journal, par défaut toutes"""
        if not self.entrées:
            imprimer_système("Votre journal est vide.")
            return
        
        entrées_à_afficher = self.entrées
        if nb_entrées:
            entrées_à_afficher = self.entrées[-nb_entrées:]
        
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ JOURNAL D'AVENTURE ═══{Couleur.RESET}")
        
        for i, entrée in enumerate(entrées_à_afficher, 1):
            importance_couleur = Couleur.BLANC
            if entrée["importance"] == 2:
                importance_couleur = Couleur.JAUNE
            elif entrée["importance"] == 3:
                importance_couleur = Couleur.ROUGE
            
            print(f"\n{importance_couleur}{Couleur.GRAS}{entrée['titre']}{Couleur.RESET}")
            print(f"{entrée['description']}")
        
        print(f"\n{Couleur.CYAN}{'═' * 30}{Couleur.RESET}")

class Inventaire:
    """Gestion des objets collectés pendant l'aventure"""
    
    def __init__(self):
        self.objets = {}
        self.capacité_max = 20
    
    def ajouter_objet(self, nom, description, quantité=1):
        """Ajoute un objet à l'inventaire ou augmente sa quantité"""
        if len(self.objets) >= self.capacité_max and nom not in self.objets:
            return False, "Inventaire plein"
            
        if nom in self.objets:
            self.objets[nom]["quantité"] += quantité
        else:
            self.objets[nom] = {
                "description": description,
                "quantité": quantité
            }
        return True, f"{nom} ajouté à l'inventaire"
    
    def retirer_objet(self, nom, quantité=1):
        """Retire une quantité d'un objet de l'inventaire"""
        if nom not in self.objets:
            return False, f"{nom} n'est pas dans l'inventaire"
        
        if self.objets[nom]["quantité"] <= quantité:
            del self.objets[nom]
        else:
            self.objets[nom]["quantité"] -= quantité
        return True, f"{quantité} {nom} retiré(s) de l'inventaire"
    
    def afficher(self):
        """Affiche le contenu de l'inventaire"""
        if not self.objets:
            imprimer_système("Votre inventaire est vide.")
            return
            
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ INVENTAIRE ═══{Couleur.RESET}")
        print(f"Objets: {len(self.objets)}/{self.capacité_max}\n")
        
        for nom, info in self.objets.items():
            print(f"{Couleur.JAUNE}{nom} ({info['quantité']}){Couleur.RESET}")
            print(f"  {info['description']}")
        
        print(f"\n{Couleur.CYAN}{'═' * 22}{Couleur.RESET}")

class Compétences:
    """Gestion des compétences du personnage"""
    
    def __init__(self):
        self.compétences = {}
    
    def ajouter_compétence(self, nom, description, niveau=1):
        """Ajoute une nouvelle compétence ou améliore une existante"""
        if nom in self.compétences:
            ancien_niveau = self.compétences[nom]["niveau"]
            self.compétences[nom]["niveau"] = max(ancien_niveau, niveau)
            if niveau > ancien_niveau:
                return True, f"{nom} amélioré au niveau {niveau}"
            return False, f"{nom} est déjà au niveau {ancien_niveau}"
        else:
            self.compétences[nom] = {
                "description": description,
                "niveau": niveau
            }
            return True, f"Nouvelle compétence acquise: {nom}"
    
    def vérifier_niveau(self, nom):
        """Vérifie si le personnage a une compétence et retourne son niveau"""
        if nom not in self.compétences:
            return 0
        return self.compétences[nom]["niveau"]
    
    def afficher(self):
        """Affiche les compétences du personnage"""
        if not self.compétences:
            imprimer_système("Vous n'avez pas encore appris de compétences.")
            return
            
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ COMPÉTENCES ═══{Couleur.RESET}\n")
        
        for nom, info in sorted(self.compétences.items(), key=lambda x: (-x[1]["niveau"], x[0])):
            niveau_txt = "★" * info["niveau"]
            print(f"{Couleur.VERT}{nom} {Couleur.JAUNE}{niveau_txt}{Couleur.RESET}")
            print(f"  {info['description']}")
        
        print(f"\n{Couleur.CYAN}{'═' * 22}{Couleur.RESET}")

class Relations:
    """Gère les relations avec les PNJ dans l'histoire"""
    
    def __init__(self):
        self.personnages = {}
    
    def ajouter_personnage(self, nom, description):
        """Ajoute un nouveau PNJ à la liste des relations"""
        if nom not in self.personnages:
            self.personnages[nom] = {
                "description": description,
                "affinité": 50,  # 0-100: hostile-amical
                "confiance": 50,  # 0-100: méfiant-confiant
                "importance": 1,  # 1-3: mineur-majeur
                "rencontres": 1,
                "notes": []
            }
    
    def modifier_relation(self, nom, affinité=0, confiance=0):
        """Modifie les scores de relation avec un PNJ"""
        if nom not in self.personnages:
            return False
            
        self.personnages[nom]["affinité"] = max(0, min(100, self.personnages[nom]["affinité"] + affinité))
        self.personnages[nom]["confiance"] = max(0, min(100, self.personnages[nom]["confiance"] + confiance))
        self.personnages[nom]["rencontres"] += 1
        return True
    
    def ajouter_note(self, nom, note):
        """Ajoute une note à propos d'un PNJ"""
        if nom in self.personnages:
            self.personnages[nom]["notes"].append({
                "texte": note,
                "date": datetime.now().isoformat()
            })
    
    def obtenir_statut_relation(self, nom):
        """Retourne une description textuelle de la relation"""
        if nom not in self.personnages:
            return "Inconnu"
            
        affinité = self.personnages[nom]["affinité"]
        confiance = self.personnages[nom]["confiance"]
        
        if affinité >= 80:
            if confiance >= 80: return "Allié proche"
            if confiance >= 50: return "Ami"
            return "Admirateur méfiant"
        elif affinité >= 50:
            if confiance >= 80: return "Confident"
            if confiance >= 50: return "Ami occasionnel"
            return "Relation cordiale"
        elif affinité >= 20:
            if confiance >= 50: return "Connaissance fiable"
            return "Simple connaissance"
        else:
            if confiance >= 50: return "Adversaire respectueux"
            if confiance >= 20: return "Rival"
            return "Ennemi"
    
    def afficher(self):
        """Affiche les relations du personnage"""
        if not self.personnages:
            imprimer_système("Vous n'avez pas encore rencontré de personnages notables.")
            return
            
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ RELATIONS ═══{Couleur.RESET}\n")
        
        for nom, info in sorted(self.personnages.items(), 
                                key=lambda x: (-x[1]["importance"], -x[1]["rencontres"])):
            statut = self.obtenir_statut_relation(nom)
            importance = "★" * info["importance"]
            
            # Déterminer la couleur en fonction de l'affinité
            if info["affinité"] >= 70:
                couleur_nom = Couleur.VERT
            elif info["affinité"] >= 40:
                couleur_nom = Couleur.JAUNE
            else:
                couleur_nom = Couleur.ROUGE
                
            print(f"{couleur_nom}{nom} {Couleur.BLANC}[{statut}] {importance}{Couleur.RESET}")
            print(f"  {info['description']}")
            
            if info["notes"]:
                print(f"  {Couleur.CYAN}Notes:{Couleur.RESET} {info['notes'][-1]['texte']}")
        
        print(f"\n{Couleur.CYAN}{'═' * 22}{Couleur.RESET}")

class SauvegardeAventure:
    """Gère la sauvegarde et le chargement des aventures"""
    
    @staticmethod
    def sauvegarder_partie(personnage, moteur, journal, inventaire, compétences, relations, nom_fichier=None):
        """Sauvegarde l'état actuel de l'aventure"""
        if not nom_fichier:
            nom_fichier = f"{personnage.nom}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # S'assurer que le dossier de sauvegarde existe
        os.makedirs("sauvegardes", exist_ok=True)
        
        # Créer le dictionnaire de sauvegarde
        données = {
            "personnage": {
                "id": personnage.id,
                "nom": personnage.nom,
                "race": personnage.race,
                "age": personnage.age,
                # Inclure tous les autres attributs...
                "description": personnage.description,
                "origine": personnage.origine,
                "lieu_origine": personnage.lieu_origine,
                "competence_initiale": personnage.competence_initiale,
                "etat_sante": personnage.etat_sante,
                "traits_personnalite": personnage.traits_personnalite,
                "niveau": personnage.niveau,
                "pv": personnage.pv,
                "pv_max": personnage.pv_max,
                "mana": personnage.mana,
                "mana_max": personnage.mana_max,
                "force": personnage.force,
                "intelligence": personnage.intelligence,
                "agilité": personnage.agilité
            },
            "moteur": {
                "dernier_segment_histoire": moteur.dernier_segment_histoire,
                "choix_en_cours": moteur.choix_en_cours,
                "scénario_progression": moteur.scénario_progression,
                "événements_passés": moteur.événements_passés,
                "objectifs_quête": moteur.objectifs_quête
            },
            "journal": {
                "entrées": journal.entrées
            },
            "inventaire": {
                "objets": inventaire.objets,
                "capacité_max": inventaire.capacité_max
            },
            "compétences": {
                "compétences": compétences.compétences
            },
            "relations": {
                "personnages": relations.personnages
            },
            "métadonnées": {
                "version": VERSION,
                "date_sauvegarde": datetime.now().isoformat(),
                "temps_jeu": moteur.temps_jeu
            }
        }
        
        # Enregistrer dans un fichier JSON
        chemin_fichier = os.path.join("sauvegardes", nom_fichier)
        try:
            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                json.dump(données, f, ensure_ascii=False, indent=2)
            return True, chemin_fichier
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False, str(e)
    
    @staticmethod
    def charger_partie(chemin_fichier):
        """Charge une partie sauvegardée à partir d'un fichier"""
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                données = json.load(f)
            
            # Créer un personnage à partir des données
            personnage = Personnage(
                id=données["personnage"]["id"],
                nom=données["personnage"]["nom"],
                race=données["personnage"]["race"]
            )
            
            # Remplir tous les attributs du personnage
            for attr, value in données["personnage"].items():
                if attr not in ["id", "nom", "race"]:  # Ces attributs sont déjà définis dans le constructeur
                    setattr(personnage, attr, value)
            
            # Créer un moteur d'histoire
            moteur = MoteurHistoire()
            moteur.dernier_segment_histoire = données["moteur"]["dernier_segment_histoire"]
            moteur.choix_en_cours = données["moteur"]["choix_en_cours"]
            moteur.scénario_progression = données["moteur"]["scénario_progression"]
            moteur.événements_passés = données["moteur"]["événements_passés"]
            moteur.objectifs_quête = données["moteur"].get("objectifs_quête", [])
            moteur.temps_jeu = données["métadonnées"].get("temps_jeu", 0)
            
            # Créer un journal
            journal = JournalAventure()
            journal.entrées = données["journal"]["entrées"]
            
            # Créer un inventaire
            inventaire = Inventaire()
            inventaire.objets = données["inventaire"]["objets"]
            inventaire.capacité_max = données["inventaire"]["capacité_max"]
            
            # Créer des compétences
            compétences = Compétences()
            compétences.compétences = données["compétences"]["compétences"]
            
            # Créer des relations
            relations = Relations()
            relations.personnages = données["relations"]["personnages"]
            
            # Créer un objet résultat pour la fonction
            résultat = {
                "personnage": personnage,
                "moteur": moteur,
                "journal": journal,
                "inventaire": inventaire,
                "compétences": compétences,
                "relations": relations,
                "métadonnées": données["métadonnées"]
            }
            
            return True, résultat
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {str(e)}")
            return False, str(e)
    
    @staticmethod
    def lister_sauvegardes():
        """Liste tous les fichiers de sauvegarde disponibles"""
        try:
            os.makedirs("sauvegardes", exist_ok=True)
            sauvegardes = []
            
            for fichier in os.listdir("sauvegardes"):
                if fichier.endswith(".json"):
                    try:
                        with open(os.path.join("sauvegardes", fichier), 'r', encoding='utf-8') as f:
                            données = json.load(f)
                            sauvegardes.append({
                                "nom_fichier": fichier,
                                "personnage": données["personnage"]["nom"],
                                "date": données["métadonnées"]["date_sauvegarde"],
                                "version": données["métadonnées"].get("version", "Inconnue"),
                                "progression": données["moteur"].get("scénario_progression", 0)
                            })
                    except Exception as e:
                        # Si le fichier est corrompu, l'ajouter quand même avec moins d'informations
                        logger.warning(f"Fichier de sauvegarde corrompu: {fichier} - {str(e)}")
                        sauvegardes.append({
                            "nom_fichier": fichier,
                            "personnage": "Inconnu",
                            "date": "Date inconnue",
                            "version": "Inconnue",
                            "progression": 0
                        })
            
            return sauvegardes
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des sauvegardes: {str(e)}")
            imprimer_erreur(f"Erreur lors de la lecture des sauvegardes: {str(e)}")
            return []

class MoteurHistoire:
    """Classe qui gère le déroulement narratif de l'histoire à choix multiples"""
    
    def __init__(self):
        self.choix_en_cours = []
        self.dernier_segment_histoire = ""
        self.fin_histoire = False
        self.temps_jeu = 0
        self.début_session = datetime.now()
        self.événements_passés = []
        self.scénario_progression = 0
        self.difficulté = 1  # 1-5: très facile à très difficile
        self.mode_ellipse = False  # Pour sauter les parties moins importantes
        self.thèmes_évités = []  # Thèmes à éviter dans la génération
        self.objectifs_quête = []  # Liste des objectifs actuels
    
    def mettre_à_jour_temps_jeu(self):
        """Met à jour le temps total de jeu"""
        maintenant = datetime.now()
        delta = (maintenant - self.début_session).total_seconds()
        self.temps_jeu += delta
        self.début_session = maintenant
    
    def ajouter_objectif(self, titre, description, priorité=1):
        """Ajoute un nouvel objectif de quête"""
        self.objectifs_quête.append({
            "titre": titre,
            "description": description,
            "priorité": priorité,  # 1-3: secondaire à principal
            "complété": False,
            "ajouté": datetime.now().isoformat()
        })
    
    def marquer_objectif_complété(self, index_ou_titre):
        """Marque un objectif comme complété"""
        if isinstance(index_ou_titre, int):
            if 0 <= index_ou_titre < len(self.objectifs_quête):
                self.objectifs_quête[index_ou_titre]["complété"] = True
                self.objectifs_quête[index_ou_titre]["date_complétion"] = datetime.now().isoformat()
                return True
        else:
            for i, obj in enumerate(self.objectifs_quête):
                if obj["titre"] == index_ou_titre and not obj["complété"]:
                    self.objectifs_quête[i]["complété"] = True
                    self.objectifs_quête[i]["date_complétion"] = datetime.now().isoformat()
                    return True
        return False
    
    def afficher_objectifs(self):
        """Affiche les objectifs actuels"""
        if not self.objectifs_quête:
            imprimer_système("Vous n'avez pas d'objectifs actuellement.")
            return
        
        objectifs_actifs = [obj for obj in self.objectifs_quête if not obj["complété"]]
        objectifs_complétés = [obj for obj in self.objectifs_quête if obj["complété"]]
        
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ OBJECTIFS DE QUÊTE ═══{Couleur.RESET}\n")
        
        if objectifs_actifs:
            print(f"{Couleur.JAUNE}● Objectifs actifs:{Couleur.RESET}")
            for i, obj in enumerate(sorted(objectifs_actifs, key=lambda x: -x["priorité"])):
                priorité_txt = "★" * obj["priorité"]
                print(f"{i+1}. {Couleur.BLANC}{obj['titre']} {Couleur.JAUNE}{priorité_txt}{Couleur.RESET}")
                print(f"   {obj['description']}")
        
        if objectifs_complétés:
            print(f"\n{Couleur.VERT}● Objectifs complétés:{Couleur.RESET}")
            for obj in objectifs_complétés[-5:]:  # Afficher les 5 derniers complétés
                print(f"✓ {Couleur.VERT}{obj['titre']}{Couleur.RESET}")
        
        print(f"\n{Couleur.CYAN}{'═' * 30}{Couleur.RESET}")

    def générer_segment_histoire(self, personnage, action_choisie=None):
        """
        Génère un segment d'histoire et des choix basés sur l'état actuel du personnage
        et l'action choisie précédemment
        """
        historique_pour_ia = config_jeu.historique[-10:] if config_jeu.historique else []
        
        # Construire le prompt système pour obtenir un segment narratif et des choix
        instruction_système = f"""Tu es le narrateur d'une aventure interactive immersive dans l'univers fantasy de Mushoku Tensei.
        L'utilisateur joue le rôle de {personnage.nom}, un(e) {personnage.race} de {personnage.age} ans.
        
        Voici des informations sur le personnage :
        - Apparence: {personnage.description}
        - Origine: {personnage.origine} de {personnage.lieu_origine}
        - Compétence principale: {personnage.competence_initiale}
        - Progression dans l'histoire: {self.scénario_progression}/100
        - Difficulté actuelle: {self.difficulté}/5
        
        INSTRUCTIONS IMPORTANTES:
        1. Raconte un segment d'histoire captivant et immersif d'environ 150 à 200 mots.
        2. Adapte ton récit à l'âge du personnage ({personnage.age} ans).
        3. Ton style doit être descriptif, riche en émotions et sensations.
        4. Ne mentionne JAMAIS que tu es une IA ou que ceci est un jeu.
        5. Termine TOUJOURS ton segment par EXACTEMENT 4 choix pour la suite.
        6. Formate tes choix comme ceci à la fin de ton texte:
        
        [Choix A] Description de la première option
        [Choix B] Description de la deuxième option
        [Choix C] Description de la troisième option
        [Choix D] Description de la quatrième option
        """
        
        if self.objectifs_quête:
            # Ajouter les objectifs actifs au prompt
            objectifs_actifs = [obj for obj in self.objectifs_quête if not obj["complété"]]
            if objectifs_actifs:
                instruction_système += "\n\nObjectifs actuels du personnage:\n"
                for obj in objectifs_actifs:
                    instruction_système += f"- {obj['titre']}: {obj['description']}\n"
        
        if self.thèmes_évités:
            instruction_système += f"\nÉvite absolument d'inclure ces thèmes: {', '.join(self.thèmes_évités)}"
        
        # Ajouter le contexte basé sur l'action précédente
        messages = [{"role": "system", "content": instruction_système}]
        
        # Ajouter l'historique des échanges
        for h in historique_pour_ia:
            messages.append({"role": h["role"], "content": h["contenu"]})
        
        # Ajouter l'action choisie si elle existe
        if action_choisie:
            messages.append({"role": "user", "content": action_choisie})
        else:
            # Premier segment de l'histoire
            age_desc = "vient de naître" if personnage.age == 0 else f"a {personnage.age} ans"
            messages.append({"role": "user", "content": 
                f"Commence mon histoire en tant que {personnage.nom} qui {age_desc}. "
                f"Décris où je me trouve et ce qui se passe autour de moi."})
        
        try:
            # Appeler l'IA
            réponse = client.chat_completion(
                model=config_jeu.modèle_ia,
                messages=messages
            )
            
            # Extraire le contenu de la réponse
            contenu_réponse = ""
            if isinstance(réponse, dict):
                if "response" in réponse and isinstance(réponse["response"], dict):
                    contenu_réponse = réponse["response"].get("content", "")
                elif "response" in réponse and isinstance(réponse["response"], str):
                    contenu_réponse = réponse["response"]
                elif "choices" in réponse:
                    contenu_réponse = réponse["choices"][0]["message"]["content"]
                else:
                    contenu_réponse = str(réponse)
            else:
                contenu_réponse = str(réponse)
            
            # Mettre à jour l'historique
            if action_choisie:
                config_jeu.historique.append({"role": "user", "contenu": action_choisie})
            else:
                config_jeu.historique.append({"role": "user", "contenu": "Commence mon histoire"})
                
            config_jeu.historique.append({"role": "assistant", "contenu": contenu_réponse})
            
            # Extraire les choix et le segment d'histoire
            segment_histoire, choix = self.extraire_choix(contenu_réponse)
            self.dernier_segment_histoire = segment_histoire
            self.choix_en_cours = choix
            
            # Ajouter cet événement à l'historique
            self.événements_passés.append({
                "segment": segment_histoire,
                "choix": choix,
                "action_choisie": action_choisie,
                "timestamp": datetime.now().isoformat()
            })
            
            # Incrémenter légèrement la progression de l'histoire
            self.scénario_progression = min(100, self.scénario_progression + random.randint(1, 3))
            
            # Si aucun choix n'est extrait, générer des choix par défaut
            if not choix or len(choix) < 4:
                self.choix_en_cours = [
                    "Explorer les environs",
                    "Parler à quelqu'un",
                    "Chercher de l'aide",
                    "Se reposer et réfléchir"
                ]
                imprimer_système("Choix générés par défaut")
            
            return segment_histoire, self.choix_en_cours
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'histoire: {str(e)}")
            imprimer_erreur(f"Erreur lors de la génération de l'histoire: {str(e)}")
            segment_erreur = "Le narrateur marque une pause, semblant réfléchir à la suite de l'histoire..."
            choix_erreur = [
                "Attendre patiemment",
                "Demander au narrateur de continuer",
                "Suggérer une direction pour l'histoire",
                "Prendre une initiative"
            ]
            self.dernier_segment_histoire = segment_erreur
            self.choix_en_cours = choix_erreur
            return segment_erreur, choix_erreur
    
    def extraire_choix(self, texte):
        """Extrait les 4 choix et le segment d'histoire du texte généré par l'IA"""
        # Rechercher les choix avec le format [Choix X]
        pattern = r"\[Choix [A-D]\].*$"
        
        # Séparer le texte en lignes
        lignes = texte.strip().split('\n')
        
        choix = []
        # Chercher les lignes qui correspondent à des choix
        for ligne in lignes:
            match = re.search(r"\[Choix ([A-D])\](.*)", ligne)
            if match:
                choix.append(match.group(2).strip())
        
        # Si le pattern des choix ne fonctionne pas, essayer un autre format
        if not choix:
            # Chercher des lignes numérotées
            for ligne in lignes:
                match = re.search(r"^[1-4][\.\)]\s*(.*)", ligne)
                if match:
                    choix.append(match.group(1).strip())
        
        # Si toujours pas de choix, chercher des lignes qui commencent par - ou *
        if not choix:
            for ligne in lignes:
                match = re.search(r"^[-*]\s*(.*)", ligne)
                if match:
                    choix.append(match.group(1).strip())
                    if len(choix) >= 4:
                        break
        
        # Pour séparer l'histoire des choix, chercher le dernier bloc de texte avant les choix
        segment_histoire = texte
        if choix:
            # Trouver l'index des premiers choix dans le texte
            choix_indices = []
            for c in choix:
                idx = texte.find(c)
                if idx != -1:
                    choix_indices.append(idx)
            
            if choix_indices:
                premier_choix_idx = min(choix_indices)
                segment_histoire = texte[:premier_choix_idx].strip()
        
        # Assurer qu'il y a exactement 4 choix
        while len(choix) < 4:
            options_défaut = [
                "Explorer les environs",
                "Parler à quelqu'un",
                "Chercher de l'aide",
                "Se reposer et réfléchir"
            ]
            choix.append(options_défaut[len(choix) % 4])
        
        return segment_histoire, choix[:4]  # Limiter à 4 choix maximum

class InterfaceAventure:
    """Gestion de l'interface utilisateur pour l'aventure à choix multiples"""
    
    def __init__(self):
        self.moteur = MoteurHistoire()
        self.journal = JournalAventure()
        self.inventaire = Inventaire()
        self.compétences = Compétences()  # Correction: Compétences au lieu de Comp
        self.relations = Relations()
        self.personnage = None
        self.derniers_choix = []
        self.première_action = True
    
    def afficher_choix_narratifs(self, choix):
        """Affiche les 4 choix narratifs de manière formatée"""
        print(f"\n{Couleur.CYAN}Que souhaitez-vous faire ?{Couleur.RESET}")
        for i, option in enumerate(choix, 1):
            print(f"{Couleur.JAUNE}{i}.{Couleur.RESET} {option}")
        print()
    
    def afficher_menu_action(self):
        """Affiche le menu d'action pendant l'aventure"""
        print(f"\n{Couleur.CYAN}═══ ACTIONS DISPONIBLES ═══{Couleur.RESET}")
        print(f"{Couleur.VERT}/journal{Couleur.RESET} - Consulter votre journal")
        print(f"{Couleur.VERT}/inventaire{Couleur.RESET} - Voir votre inventaire")
        print(f"{Couleur.VERT}/compétences{Couleur.RESET} - Voir vos compétences")
        print(f"{Couleur.VERT}/relations{Couleur.RESET} - Voir vos relations")
        print(f"{Couleur.VERT}/objectifs{Couleur.RESET} - Voir vos objectifs actuels")
        print(f"{Couleur.VERT}/statut{Couleur.RESET} - Voir le statut de votre personnage")
        print(f"{Couleur.VERT}/sauvegarder{Couleur.RESET} - Sauvegarder l'aventure")
        print(f"{Couleur.VERT}/quitter{Couleur.RESET} - Quitter l'aventure")
        print(f"{Couleur.CYAN}{'═' * 26}{Couleur.RESET}")
        print("\nEntrez un numéro de choix (1-4) ou une commande:")
    
    def traiter_commande(self, commande):
        """Traite les commandes spéciales pendant l'aventure"""
        if commande.lower() in ["/journal", "/j"]:
            self.journal.afficher()
            return True
            
        elif commande.lower() in ["/inventaire", "/inv", "/i"]:
            self.inventaire.afficher()
            return True
            
        elif commande.lower() in ["/compétences", "/comp", "/c"]:
            self.compétences.afficher()
            return True
            
        elif commande.lower() in ["/relations", "/rel", "/r"]:
            self.relations.afficher()
            return True
            
        elif commande.lower() in ["/objectifs", "/obj", "/o"]:
            self.moteur.afficher_objectifs()
            return True
            
        elif commande.lower() in ["/statut", "/stats", "/s"]:
            self.afficher_statut_personnage()
            return True
            
        elif commande.lower() in ["/sauvegarder", "/save"]:
            success, résultat = SauvegardeAventure.sauvegarder_partie(
                self.personnage, self.moteur, self.journal, 
                self.inventaire, self.compétences, self.relations
            )
            
            if success:
                imprimer_système(f"Partie sauvegardée dans {résultat}")
            else:
                imprimer_erreur(f"Erreur lors de la sauvegarde: {résultat}")
            return True
            
        elif commande.lower() in ["/aide", "/help", "/h"]:
            self.afficher_menu_action()
            return True
            
        elif commande.lower() in ["/quitter", "/exit", "/q"]:
            if input(f"{Couleur.JAUNE}Voulez-vous vraiment quitter ? (o/n) {Couleur.RESET}").lower() == "o":
                self.moteur.fin_histoire = True
            return True
            
        # Si ce n'est pas une commande reconnue
        return False
    
    def afficher_statut_personnage(self):
        """Affiche les statistiques du personnage"""
        if not self.personnage:
            imprimer_erreur("Aucun personnage actif.")
            return
            
        p = self.personnage
        
        print(f"\n{Couleur.CYAN}{Couleur.GRAS}═══ STATUT DU PERSONNAGE ═══{Couleur.RESET}\n")
        print(f"{Couleur.JAUNE}{Couleur.GRAS}{p.nom}{Couleur.RESET}, {p.race} de {p.age} ans")
        print(f"Origine: {p.origine} de {p.lieu_origine}")
        print(f"Compétence initiale: {p.competence_initiale}")
        print(f"Santé: {p.etat_sante}")
        
        print(f"\n{Couleur.VERT}● Apparence:{Couleur.RESET}")
        print(f"  {p.description}")
        
        # Afficher les statistiques
        print(f"\n{Couleur.JAUNE}● Statistiques:{Couleur.RESET}")
        print(f"  Niveau: {p.niveau}")
        print(f"  PV: {p.pv}/{p.pv_max}")
        print(f"  Mana: {p.mana}/{p.mana_max}")
        print(f"  Force: {p.force}")
        print(f"  Intelligence: {p.intelligence}")
        print(f"  Agilité: {p.agilité}")
        
        # Afficher les traits de personnalité
        print(f"\n{Couleur.MAGENTA}● Traits de personnalité:{Couleur.RESET}")
        for trait, valeur in p.traits_personnalite.items():
            barre = "▰" * (valeur // 10) + "▱" * (10 - (valeur // 10))
            print(f"  {trait.capitalize()}: {barre} ({valeur}/100)")
        
        # Progression de l'histoire
        print(f"\n{Couleur.CYAN}● Progression:{Couleur.RESET}")
        print(f"  Avancement: {self.moteur.scénario_progression}%")
        print(f"  Temps de jeu: {self.moteur.temps_jeu//3600}h {(self.moteur.temps_jeu%3600)//60}m")
        
        print(f"\n{Couleur.CYAN}{'═' * 32}{Couleur.RESET}")
    
    def démarrer_aventure(self):
        """Démarre et gère une aventure narrative à choix multiples"""
        # Créer un personnage
        self.personnage = créer_nouveau_personnage()
        
        if not self.personnage:
            imprimer_erreur("Impossible de créer un personnage. Retour au menu principal.")
            return
        
        effacer_écran()
        imprimer_titre()
        
        age_description = "vient de naître" if self.personnage.age == 0 else f"est âgé(e) de {self.personnage.age} ans"
        imprimer_narration(f"L'histoire de {self.personnage.nom}, qui {age_description}, commence dans le monde de Mushoku Tensei...")
        time.sleep(2)
        
        # Initialiser le moteur avec ce personnage
        action_choisie = None
        
        # Boucle principale de l'aventure
        while not self.moteur.fin_histoire:
            self.moteur.mettre_à_jour_temps_jeu()
            
            effacer_écran()
            imprimer_titre()
            
            # Générer le segment d'histoire et les choix
            segment, choix = self.moteur.générer_segment_histoire(self.personnage, action_choisie)
            
            # Afficher le segment d'histoire
            imprimer_séparateur()
            imprimer_narration(segment)
            imprimer_séparateur()
            
            # Afficher les choix
            self.afficher_choix_narratifs(choix)
            
            if self.première_action:
                self.afficher_menu_action()
                self.première_action = False
            
            # Obtenir le choix du joueur
            choix_valide = False
            while not choix_valide:
                entrée = input(f"{Couleur.VERT}> {Couleur.RESET}")
                
                # Vérifier si c'est une commande spéciale
                if entrée.startswith("/"):
                    commande_traitée = self.traiter_commande(entrée)
                    if commande_traitée:
                        # Si la commande signale la fin
                        if self.moteur.fin_histoire:
                            return
                        
                        # Sinon, continuer à attendre un choix valide
                        input(f"\n{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")
                        effacer_écran()
                        imprimer_titre()
                        imprimer_séparateur()
                        imprimer_narration(segment)
                        imprimer_séparateur()
                        self.afficher_choix_narratifs(choix)
                        continue
                
                # Traiter comme un choix numérique
                if entrée.isdigit():
                    choix_num = int(entrée)
                    if 1 <= choix_num <= len(choix):
                        action_choisie = choix[choix_num - 1]
                        self.derniers_choix.append(action_choisie)
                        
                        # Enregistrer ce choix dans le journal si c'est significatif
                        if self.est_choix_significatif(action_choisie):
                            self.journal.ajouter_entrée(
                                f"Décision: {action_choisie[:30]}...",
                                f"Vous avez choisi: {action_choisie}\nContexte: {segment[-100:] if len(segment) > 100 else segment}",
                                importance=2
                            )
                        
                        choix_valide = True
                    else:
                        imprimer_erreur(f"Veuillez entrer un nombre entre 1 et {len(choix)}.")
                else:
                    # Si l'entrée n'est pas un nombre et pas une commande reconnue
                    imprimer_erreur("Veuillez entrer un nombre valide ou une commande commençant par /")
        
        # Fin de l'aventure
        self.moteur.mettre_à_jour_temps_jeu()
        imprimer_système(f"Aventure terminée! Temps total de jeu: {self.moteur.temps_jeu//3600}h {(self.moteur.temps_jeu%3600)//60}m")
        input(f"{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")
    
    def est_choix_significatif(self, choix):
        """Détermine si un choix est significatif pour l'histoire"""
        mots_clés_significatifs = [
            "décider", "choisir", "accepter", "refuser", "tuer", "sauver",
            "alliance", "trahir", "révéler", "cacher", "combattre", "fuir"
        ]
        
        # Vérifier si un des mots clés est présent dans le choix
        return any(mot in choix.lower() for mot in mots_clés_significatifs)


def aventure_interactive():
    """Fonction principale qui lance l'aventure à choix multiples"""
    interface = InterfaceAventure()
    interface.démarrer_aventure()


# Interface pour charger une partie sauvegardée
def charger_aventure():
    """Interface pour charger une partie sauvegardée"""
    effacer_écran()
    imprimer_titre()
    print(f"{Couleur.CYAN}╔════════════ CHARGEMENT D'UNE PARTIE ════════════╗{Couleur.RESET}")
    
    # Récupérer la liste des sauvegardes
    sauvegardes = SauvegardeAventure.lister_sauvegardes()
    
    if not sauvegardes:
        imprimer_erreur("Aucune sauvegarde trouvée!")
        input(f"{Couleur.JAUNE}Appuyez sur Entrée pour revenir au menu...{Couleur.RESET}")
        return False
    
    # Afficher les sauvegardes disponibles
    print(f"\n{Couleur.JAUNE}Sauvegardes disponibles:{Couleur.RESET}\n")
    
    for i, sauvegarde in enumerate(sauvegardes, 1):
        date_formattée = sauvegarde["date"]
        try:
            # Convertir le format ISO en datetime puis en format plus lisible
            date_obj = datetime.fromisoformat(sauvegarde["date"].replace('Z', '+00:00'))
            date_formattée = date_obj.strftime("%d/%m/%Y %H:%M")
        except:
            # Garder le format original en cas d'erreur
            pass
            
        print(f"{i}. {sauvegarde['personnage']} - {date_formattée} (Progression: {sauvegarde['progression']}%)")
    
    print(f"\n{len(sauvegardes) + 1}. Retour")
    
    # Demander au joueur de choisir une sauvegarde
    choix = 0
    while not (1 <= choix <= len(sauvegardes) + 1):
        try:
            choix = int(input(f"\n{Couleur.JAUNE}Choisissez une sauvegarde (1-{len(sauvegardes) + 1}): {Couleur.RESET}"))
        except ValueError:
            imprimer_erreur("Veuillez entrer un nombre valide.")
    
    # Option de retour
    if choix == len(sauvegardes) + 1:
        return False
    
    # Charger la sauvegarde sélectionnée
    sauvegarde_choisie = sauvegardes[choix - 1]
    
    imprimer_système(f"Chargement de la sauvegarde de {sauvegarde_choisie['personnage']}...")
    
    success, résultat = SauvegardeAventure.charger_partie(os.path.join("sauvegardes", sauvegarde_choisie["nom_fichier"]))
    
    if not success:
        imprimer_erreur(f"Erreur lors du chargement: {résultat}")
        input(f"{Couleur.JAUNE}Appuyez sur Entrée pour revenir au menu...{Couleur.RESET}")
        return False
    
    # Créer une interface avec les données chargées
    interface = InterfaceAventure()
    interface.personnage = résultat["personnage"]
    interface.moteur = résultat["moteur"]
    interface.journal = résultat["journal"]
    interface.inventaire = résultat["inventaire"]
    interface.compétences = résultat["compétences"]
    interface.relations = résultat["relations"]
    
    # Démarrer l'aventure à partir du dernier point
    imprimer_système(f"Sauvegarde chargée avec succès! Reprise de l'aventure...")
    time.sleep(1)
    
    # Boucle principale de l'aventure (code identique à démarrer_aventure mais sans la création de personnage)
    action_choisie = None
    
    while not interface.moteur.fin_histoire:
        interface.moteur.mettre_à_jour_temps_jeu()
        
        effacer_écran()
        imprimer_titre()
        
        # Générer le segment d'histoire et les choix
        segment, choix = interface.moteur.générer_segment_histoire(interface.personnage, action_choisie)
        
        # Afficher le segment d'histoire
        imprimer_séparateur()
        imprimer_narration(segment)
        imprimer_séparateur()
        
        # Afficher les choix
        interface.afficher_choix_narratifs(choix)
        
        # Obtenir le choix du joueur
        choix_valide = False
        while not choix_valide:
            entrée = input(f"{Couleur.VERT}> {Couleur.RESET}")
            
            # Vérifier si c'est une commande spéciale
            if entrée.startswith("/"):
                commande_traitée = interface.traiter_commande(entrée)
                if commande_traitée:
                    # Si la commande signale la fin
                    if interface.moteur.fin_histoire:
                        return True
                    
                    # Sinon, continuer à attendre un choix valide
                    input(f"\n{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")
                    effacer_écran()
                    imprimer_titre()
                    imprimer_séparateur()
                    imprimer_narration(segment)
                    imprimer_séparateur()
                    interface.afficher_choix_narratifs(choix)
                    continue
            
            # Traiter comme un choix numérique
            if entrée.isdigit():
                choix_num = int(entrée)
                if 1 <= choix_num <= len(choix):
                    action_choisie = choix[choix_num - 1]
                    interface.derniers_choix.append(action_choisie)
                    
                    # Enregistrer ce choix dans le journal si c'est significatif
                    if interface.est_choix_significatif(action_choisie):
                        interface.journal.ajouter_entrée(
                            f"Décision: {action_choisie[:30]}...",
                            f"Vous avez choisi: {action_choisie}\nContexte: {segment[-100:] if len(segment) > 100 else segment}",
                            importance=2
                        )
                    
                    choix_valide = True
                else:
                    imprimer_erreur(f"Veuillez entrer un nombre entre 1 et {len(choix)}.")
            else:
                # Si l'entrée n'est pas un nombre et pas une commande reconnue
                imprimer_erreur("Veuillez entrer un nombre valide ou une commande commençant par /")
    
    # Fin de l'aventure
    interface.moteur.mettre_à_jour_temps_jeu()
    imprimer_système(f"Aventure terminée! Temps total de jeu: {interface.moteur.temps_jeu//3600}h {(interface.moteur.temps_jeu%3600)//60}m")
    input(f"{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")
    return True


# Menu principal autonome
def menu_principal():
    """Affiche le menu principal du module d'aventure à choix multiples"""
    while True:
        effacer_écran()
        imprimer_titre()
        
        print(f"{Couleur.CYAN}╔═══════════════════ MENU PRINCIPAL ═══════════════════╗{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 1. Nouvelle Aventure                                 {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 2. Charger une Aventure                              {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 3. Options                                           {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}║{Couleur.RESET} 4. Quitter                                           {Couleur.CYAN}║{Couleur.RESET}")
        print(f"{Couleur.CYAN}╚═══════════════════════════════════════════════════════╝{Couleur.RESET}")
        
        choix = input(f"{Couleur.JAUNE}Votre choix: {Couleur.RESET}")
        
        if choix == "1":
            aventure_interactive()
        elif choix == "2":
            charger_aventure()
        elif choix == "3":
            # Options - À implémenter
            imprimer_système("Options non disponibles dans cette version.")
            input(f"{Couleur.JAUNE}Appuyez sur Entrée pour continuer...{Couleur.RESET}")
        elif choix == "4":
            imprimer_système("Au revoir!")
            return
        else:
            imprimer_erreur("Option invalide.")
            time.sleep(1)


# Point d'entrée principal
if __name__ == "__main__":
    try:
        # En mode direct, afficher un message explicatif et démarrer le menu
        print(f"{Couleur.VERT}=" * 60)
        print(f"Démarrage du mode Aventure à Choix Multiples de Mushoku Tensei RP")
        print(f"Version: {VERSION}")
        print(f"=" * 60 + f"{Couleur.RESET}\n")
        
        menu_principal()
        
    except KeyboardInterrupt:
        print("\nArrêt du programme par l'utilisateur.")
    except Exception as e:
        print(f"\n{Couleur.ROUGE}Erreur fatale: {str(e)}{Couleur.RESET}")
        if hasattr(config_jeu, 'mode_debug') and config_jeu.mode_debug:
            import traceback
            traceback.print_exc()