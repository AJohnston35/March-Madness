import sys
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QPushButton, QFrame, QGridLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPainter, QFont, QIcon
from PyQt5.QtCore import Qt, QSize, QRect
import joblib

from helper import get_data, preprocess_data  # Import your data functions

class TeamStats:
    def __init__(self, team='', year='', conference='', wins=0, losses=0, points=0.0, 
                opp_points=0.0, margin_of_victory=0.0, strength_of_schedule=0.0, 
                offensive_srs=0.0, defensive_srs=0.0, simple_rating_system=0.0,
                offensive_rating=0.0, defensive_rating=0.0, net_rating=0.0):
        self.team = team
        self.year = year
        self.conference = conference
        self.wins = wins
        self.losses = losses
        self.points = points
        self.opp_points = opp_points
        self.margin_of_victory = margin_of_victory
        self.strength_of_schedule = strength_of_schedule
        self.offensive_srs = offensive_srs
        self.defensive_srs = defensive_srs
        self.simple_rating_system = simple_rating_system
        self.offensive_rating = offensive_rating
        self.defensive_rating = defensive_rating
        self.net_rating = net_rating

class TeamSideWidget(QWidget):
    def __init__(self, is_left, parent=None):
        super().__init__(parent)
        self.is_left = is_left
        self.parent_app = parent
        
        # Set side colors
        self.side_colors = (QColor("#2E2C2B"), QColor("#000000")) if is_left else (QColor("#6A6260"), QColor("#86807F"))
        
        # Main layout with more spacing for centered appearance
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 30, 20, 30)  # Add more margin around elements
        self.layout.setSpacing(20)  # Increase spacing between elements
        self.setLayout(self.layout)
        
        # Year selector
        year_layout = QHBoxLayout()
        year_label = QLabel("Year:")
        year_label.setStyleSheet("background-color: transparent; color: white; font-weight: bold;")
        self.year_combo = QComboBox()
        self.year_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: #333333;
                border: 1px solid white;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: #333333;
            }
        """)

        year_layout.addWidget(year_label)
        year_layout.addWidget(self.year_combo)
        year_layout.addStretch()
        self.layout.addLayout(year_layout)
        
        # Add stretch to push logo to center vertically
        self.layout.addStretch(1)
        
        # Team logo container - make it larger
        logo_container = QHBoxLayout()
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMinimumSize(350, 350)  # Increased size for larger logos
        self.logo_label.setStyleSheet("background-color: transparent;")
        
        logo_container.addStretch(1)
        logo_container.addWidget(self.logo_label)
        logo_container.addStretch(1)
        
        self.layout.addLayout(logo_container)
        
        # Add stretch to push logo to center vertically
        self.layout.addStretch(1)
        
        # Team selector
        self.team_combo = QComboBox()
        self.team_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: #333333;
                border: 1px solid white;
                padding: 5px;
                font-size: 16px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: #333333;
            }
        """)
        self.layout.addWidget(self.team_combo)
        
        # Connect signals
        self.year_combo.currentTextChanged.connect(self.year_changed)
        self.team_combo.currentTextChanged.connect(self.team_changed)
        
    def year_changed(self, year):
        if not year:
            return
            
        # Update available teams for the selected year
        if self.parent_app:
            teams = self.parent_app.get_teams_for_year(year)
            current_team = self.team_combo.currentText()
            
            # Block signals to prevent triggering team_changed while updating
            self.team_combo.blockSignals(True)
            
            self.team_combo.clear()
            self.team_combo.addItems(teams)
            
            # Try to keep the same team if it exists in new year
            if current_team in teams:
                index = self.team_combo.findText(current_team)
                self.team_combo.setCurrentIndex(index)
            
            self.team_combo.blockSignals(False)
            
            # Manually trigger update if team changed
            if self.is_left:
                self.parent_app.left_year = year
            else:
                self.parent_app.right_year = year
                
            # Update stats and UI
            self.parent_app.update_team_stats()
            
    def team_changed(self, team):
        if not team:
            return
            
        if self.is_left:
            self.parent_app.left_team = team
        else:
            self.parent_app.right_team = team
            
        # Update team logo
        self.update_logo(team)
        
        # Update stats and UI
        self.parent_app.update_team_stats()
            
    def update_logo(self, team):
        if not team:
            # If no team selected, show placeholder
            self.logo_label.setText(team[0] if team else "")
            self.logo_label.setStyleSheet("color: white; font-size: 180px; font-weight: bold; background-color: transparent;")
            return
            
        # Normalize team name to match filename format (same as Flutter code)
        normalized_name = self.parent_app.normalize_team_name(team)
        logo_path = os.path.join( "assets", "logos", f"{normalized_name}.png")
        
        # Try to load the image, use placeholder if not found
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            # Scale to larger size for better visibility (400x400)
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setText("")
        else:
            # Use first letter as placeholder with larger font
            self.logo_label.setText(team[0] if team else "")
            self.logo_label.setStyleSheet("color: white; font-size: 180px; font-weight: bold; background-color: transparent;")
    
    def paintEvent(self, event):
        # Create gradient background
        painter = QPainter(self)
        gradient = QLinearGradient()
        
        if self.is_left:
            gradient.setStart(self.width(), self.height() / 2)
            gradient.setFinalStop(0, self.height() / 2)
        else:
            gradient.setStart(0, self.height() / 2)
            gradient.setFinalStop(self.width(), self.height() / 2)
            
        gradient.setColorAt(0, self.side_colors[0])
        gradient.setColorAt(1, self.side_colors[1])
        
        painter.fillRect(self.rect(), gradient)
        
class StatsRow(QWidget):
    def __init__(self, label, left_value, right_value, left_is_better, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Left value
        self.left_label = QLabel(left_value)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet(f"""
            color: white;
            font-weight: {'bold' if left_is_better else 'normal'};
            background-color: {f'rgba(0, 128, 0, 0.3)' if left_is_better else 'transparent'};
            padding: 4px;
        """)
        
        # Center label
        center_label = QLabel(label)
        center_label.setAlignment(Qt.AlignCenter)
        center_label.setFixedWidth(140)
        center_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        
        # Right value
        self.right_label = QLabel(right_value)
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet(f"""
            color: white;
            font-weight: {'bold' if not left_is_better else 'normal'};
            background-color: {f'rgba(0, 128, 0, 0.3)' if not left_is_better else 'transparent'};
            padding: 4px;
        """)
        
        layout.addWidget(self.left_label, 1)
        layout.addWidget(center_label)
        layout.addWidget(self.right_label, 1)

class NCAATeamMatchupApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Prediction model
        self.lgbm_model = joblib.load('lgbm_model.joblib')
        
        # App settings
        self.setWindowTitle("NCAA Team Matchup")
        self.setMinimumSize(1024, 768)
        
        # Initialize data
        self.left_team = ""
        self.right_team = ""
        self.left_year = "2024"
        self.right_year = "2024"
        self.selected_round = "First Round"
        
        self.left_team_stats = TeamStats()
        self.right_team_stats = TeamStats()
        
        self.all_teams = []
        self.teams_by_year = {}
        self.all_team_stats = {}
        
        self.available_years = [
            "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", 
            "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008", "2007", "2006", 
            "2005", "2004", "2003", "2002", "2001", "2000", "1999", "1998", "1997", "1996", 
            "1995", "1994", "1993", "1992", "1991", "1990", "1989", "1988", "1987", "1986", "1985"
        ]
        
        self.tournament_rounds = [
            "First Round", "Second Round", "Sweet Sixteen", "Elite Eight", "Final Four", "Championship"
        ]
        
        # Set up UI
        self.init_ui()
        
        # Load the data
        self.load_teams_from_csv()
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create header
        header = QFrame()
        header.setStyleSheet("background-color: black;")
        header_layout = QHBoxLayout()
        header.setLayout(header_layout)
        header.setFixedHeight(80)
        
        # Logo and title
        logo_label = QLabel()
        logo_pixmap = QPixmap("assets/logos/March_Madness_logo.png")
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(100, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        
        title_label = QLabel("Matchup Prediction")
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")
        
        # Tournament round selector
        round_layout = QHBoxLayout()
        round_label = QLabel("Tournament Round:")
        round_label.setStyleSheet("color: white; font-size: 16px;")
        
        self.round_combo = QComboBox()
        self.round_combo.addItems(self.tournament_rounds)
        self.round_combo.setCurrentText(self.selected_round)
        self.round_combo.setStyleSheet("""
            QComboBox {
                color: white;
                background-color: black;
                border-bottom: 2px solid orange;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: black;
            }
        """)
        
        round_layout.addWidget(round_label)
        round_layout.addWidget(self.round_combo)
        
        # Add items to header
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(round_layout)
        
        # Team comparison area
        teams_layout = QHBoxLayout()
        
        # Left team side
        self.left_team_widget = TeamSideWidget(is_left=True, parent=self)
        
        # Right team side
        self.right_team_widget = TeamSideWidget(is_left=False, parent=self)
        
        teams_layout.addWidget(self.left_team_widget)
        teams_layout.addWidget(self.right_team_widget)
        
        # Stats comparison section - improve to match screenshot
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: black; border: none;")
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding inside
        stats_frame.setLayout(stats_layout)
        
        stats_title = QLabel("TEAM STATS COMPARISON")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_title.setStyleSheet("color: white; font-weight: bold; font-size: 16px; padding: 6px;")
        stats_layout.addWidget(stats_title)
        
        # Stats rows will be added dynamically after data loads
        self.stats_grid = QVBoxLayout()
        self.stats_grid.setSpacing(8)  # Add more space between rows
        stats_layout.addLayout(self.stats_grid)
        
        # Predict button - improve styling to match the screenshot
        predict_button = QPushButton("PREDICT WINNER")
        predict_button.setFixedSize(270, 50)
        predict_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: black;
                border: 2px solid #444;
                border-radius: 25px;
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 0;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #222;
            }
            QPushButton:pressed {
                background-color: #111;
            }
        """)
        predict_button_layout = QHBoxLayout()
        predict_button_layout.setContentsMargins(0, 10, 0, 10)  # Reduce vertical margins
        predict_button_layout.addStretch()
        predict_button.clicked.connect(self.predict_winner)
        predict_button_layout.addWidget(predict_button)
        predict_button_layout.addStretch()
        
        # Add everything to main layout - remove the bottom spacing
        main_layout.addWidget(header)
        main_layout.addLayout(teams_layout, 1)
        main_layout.addWidget(stats_frame)
        main_layout.addLayout(predict_button_layout)
        
        # Remove any spacing from the bottom of the window
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("QMainWindow {background-color: black; margin: 0; padding: 0;}")
        main_widget.setStyleSheet("QWidget {background-color: black; margin: 0; padding: 0;}")
    
    def load_teams_from_csv(self):
        try:
            # Open and read the CSV file
            with open('team_ratings/all_ratings.csv', 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                # Skip header
                header = next(reader)
                
                # Initialize data structures
                unique_teams = set()
                teams_by_year = {}
                all_team_stats = {}
                
                # Process each row
                for row in reader:
                    if len(row) < 15:
                        print(f"Skipping row: insufficient columns: {len(row)}")
                        continue
                    
                    team_name = row[1]
                    year = row[15]
                    
                    if not team_name or not year:
                        print("Skipping row: empty team or year")
                        continue
                    
                    unique_teams.add(team_name)
                    
                    # Store teams by year
                    if year not in teams_by_year:
                        teams_by_year[year] = set()
                        all_team_stats[year] = {}
                    
                    teams_by_year[year].add(team_name)
                    
                    # Create TeamStats object
                    all_team_stats[year][team_name] = TeamStats(
                        team=team_name,
                        year=year,
                        conference=row[2].strip(),
                        wins=self.parse_int_safe(row[3]),
                        losses=self.parse_int_safe(row[4]),
                        points=self.parse_float_safe(row[5]),
                        opp_points=self.parse_float_safe(row[6]),
                        margin_of_victory=self.parse_float_safe(row[7]),
                        strength_of_schedule=self.parse_float_safe(row[8]),
                        offensive_srs=self.parse_float_safe(row[9]),
                        defensive_srs=self.parse_float_safe(row[10]),
                        simple_rating_system=self.parse_float_safe(row[11]),
                        offensive_rating=self.parse_float_safe(row[12]),
                        defensive_rating=self.parse_float_safe(row[13]),
                        net_rating=self.parse_float_safe(row[12]) - self.parse_float_safe(row[13])
                    )
                # Convert sets to sorted lists
                sorted_teams_by_year = {}
                for year, teams in teams_by_year.items():
                    sorted_teams_by_year[year] = sorted(list(teams))
                
                # Store data
                self.all_teams = sorted(list(unique_teams))
                self.teams_by_year = sorted_teams_by_year
                self.all_team_stats = all_team_stats
                
                # Update UI with data
                self.update_ui_with_data()
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    def parse_int_safe(self, value):
        if not value:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def parse_float_safe(self, value):
        if not value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_teams_for_year(self, year):
        return self.teams_by_year.get(year, self.all_teams)
    
    def update_ui_with_data(self):
        # Update year dropdowns
        self.left_team_widget.year_combo.addItems(self.available_years)
        self.right_team_widget.year_combo.addItems(self.available_years)
        
        # Set default years
        self.left_team_widget.year_combo.setCurrentText(self.left_year)
        self.right_team_widget.year_combo.setCurrentText(self.right_year)
        
        # Update team dropdowns with teams for selected years
        left_teams = self.get_teams_for_year(self.left_year)
        right_teams = self.get_teams_for_year(self.right_year)
        
        self.left_team_widget.team_combo.clear()
        self.right_team_widget.team_combo.clear()
        
        self.left_team_widget.team_combo.addItems(left_teams)
        self.right_team_widget.team_combo.addItems(right_teams)
        
        # Set default teams if available
        if left_teams:
            self.left_team = left_teams[0]
            self.left_team_widget.team_combo.setCurrentText(self.left_team)
            self.left_team_widget.update_logo(self.left_team)
            
        if right_teams:
            self.right_team = right_teams[0]
            self.right_team_widget.team_combo.setCurrentText(self.right_team)
            self.right_team_widget.update_logo(self.right_team)
        
        # Update team stats
        self.update_team_stats()
    
    def update_team_stats(self):
        print(f"Updating team stats for: {self.left_team} ({self.left_year}) and {self.right_team} ({self.right_year})")
        
        # Debug info
        print(f"Available years in data: {', '.join(self.all_team_stats.keys())}")
        
        # Check if selected years exist in data
        if self.left_year not in self.all_team_stats:
            print(f"WARNING: Year {self.left_year} not found in data")
        if self.right_year not in self.all_team_stats:
            print(f"WARNING: Year {self.right_year} not found in data")
        
        # Update left team stats
        if (self.left_team and self.left_year in self.all_team_stats and 
            self.left_team in self.all_team_stats[self.left_year]):
            self.left_team_stats = self.all_team_stats[self.left_year][self.left_team]
            print(f"Found stats for {self.left_team} in {self.left_year}: W-L {self.left_team_stats.wins}-{self.left_team_stats.losses}")
        else:
            print(f"No stats found for {self.left_team} in {self.left_year} - using empty stats")
            self.left_team_stats = TeamStats()
        
        # Update right team stats
        if (self.right_team and self.right_year in self.all_team_stats and 
            self.right_team in self.all_team_stats[self.right_year]):
            self.right_team_stats = self.all_team_stats[self.right_year][self.right_team]
            print(f"Found stats for {self.right_team} in {self.right_year}: W-L {self.right_team_stats.wins}-{self.right_team_stats.losses}")
        else:
            print(f"No stats found for {self.right_team} in {self.right_year} - using empty stats")
            self.right_team_stats = TeamStats()
        
        # Update stats comparison UI
        self.update_stats_comparison()
    
    def update_stats_comparison(self):
        # Clear existing stats rows
        for i in reversed(range(self.stats_grid.count())):
            widget = self.stats_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Helper function to calculate win percentage
        def safe_win_percentage(stats):
            total_games = stats.wins + stats.losses
            if total_games == 0:
                return 0.0
            return stats.wins / total_games
        
        # Add new stat rows
        # Record
        left_is_better = safe_win_percentage(self.left_team_stats) > safe_win_percentage(self.right_team_stats)
        record_row = StatsRow(
            "Record", 
            f"{self.left_team_stats.wins}-{self.left_team_stats.losses}", 
            f"{self.right_team_stats.wins}-{self.right_team_stats.losses}",
            left_is_better
        )
        self.stats_grid.addWidget(record_row)
        
        # Points Per Game
        ppg_row = StatsRow(
            "Points Per Game",
            f"{self.left_team_stats.points:.1f}",
            f"{self.right_team_stats.points:.1f}",
            self.left_team_stats.points > self.right_team_stats.points
        )
        self.stats_grid.addWidget(ppg_row)
        
        # Opponent PPG
        opp_ppg_row = StatsRow(
            "Opponent PPG",
            f"{self.left_team_stats.opp_points:.1f}",
            f"{self.right_team_stats.opp_points:.1f}",
            self.left_team_stats.opp_points < self.right_team_stats.opp_points
        )
        self.stats_grid.addWidget(opp_ppg_row)
        
        # Margin of Victory
        mov_row = StatsRow(
            "Margin of Victory",
            f"{self.left_team_stats.margin_of_victory:.1f}",
            f"{self.right_team_stats.margin_of_victory:.1f}",
            self.left_team_stats.margin_of_victory > self.right_team_stats.margin_of_victory
        )
        self.stats_grid.addWidget(mov_row)
        
        # Strength of Schedule
        sos_row = StatsRow(
            "Strength of Schedule",
            f"{self.left_team_stats.strength_of_schedule:.2f}",
            f"{self.right_team_stats.strength_of_schedule:.2f}",
            self.left_team_stats.strength_of_schedule > self.right_team_stats.strength_of_schedule
        )
        self.stats_grid.addWidget(sos_row)
        
        # Simple Rating System
        srs_row = StatsRow(
            "Rating (SRS)",
            f"{self.left_team_stats.simple_rating_system:.2f}",
            f"{self.right_team_stats.simple_rating_system:.2f}",
            self.left_team_stats.simple_rating_system > self.right_team_stats.simple_rating_system
        )
        self.stats_grid.addWidget(srs_row)
        
        # Net Rating
        net_row = StatsRow(
            "Net Rating",
            f"{self.left_team_stats.net_rating:.2f}",
            f"{self.right_team_stats.net_rating:.2f}",
            self.left_team_stats.net_rating > self.right_team_stats.net_rating
        )
        self.stats_grid.addWidget(net_row)
    
    def normalize_team_name(self, team_name):
        # Map of team name variations to their standardized filename
        team_name_map = {
            'Abilene Christian': 'abilene_christian_wildcats',
            'Air Force': 'air_force_falcons',
            'Akron': 'akron_zips',
            'Alabama': 'alabama_crimson_tide',
            'Alabama A&M': 'alabama_am_bulldogs',
            'UAB': 'alabama_birmingham_blazers',
            'Alabama State': 'alabama_state_hornets',
            'Albany': 'albany_great_danes',
            'Alcorn State': 'alcorn_state_braves',
            'American': 'american_eagles',
            'America East': 'appalachian_state_mountaineers',
            'Appalachian State': 'appalachian_state_mountaineers',
            'Arizona State': 'arizona_state_sun_devils',
            'Arizona': 'arizona_wildcats',
            'Little Rock': 'arkansas_little_rock_trojans',
            'Arkansas Pine Bluff': 'arkansas_pine_bluff_golden_lions',
            'Arkansas': 'arkansas_razorbacks',
            'Arkansas State': 'arkansas_state_red_wolves',
            'Army': 'army_west_point_black_knights',
            'Atlantic 10': 'atlantic_10_conference',
            'Atlantic Coast': 'atlantic_coast_conference_acc',
            'Atlantic Sun': 'atlantic_sun_conference_asun',
            'Auburn': 'auburn_tigers',
            'Austin Peay': 'austin_peay_governors',
            'Ball State': 'ball_state_cardinals',
            'Baylor': 'baylor_bears',
            'Bellarmine': 'bellarmine_knights',
            'Belmont': 'belmont_bruins',
            'Bethune-Cookman': 'bethune_cookman_wildcats',
            'Big 12': 'big_12_conference',
            'Big East': 'big_east_conference',
            'Big Sky': 'big_sky_conference',
            'Big South': 'big_south_conference',
            'Big Ten': 'big_ten_conference',
            'Big West': 'big_west_conference',
            'Binghamton': 'binghamton_bearcats',
            'Boise State': 'boise_state_broncos',
            'Boston College': 'boston_college_eagles',
            'Boston University': 'boston_university_terriers',
            'Bowling Green': 'bowling_green_falcons',
            'Bradley': 'bradley_braves',
            'Bryant': 'bryant_bulldogs',
            'Bucknell': 'bucknell_bison',
            'Buffalo': 'buffalo_bulls',
            'Butler': 'butler_bulldogs',
            'BYU': 'byu_cougars',
            'California Baptist': 'california_baptist_lancers',
            'California': 'california_golden_bears',
            'Long Beach State': 'cal_long_beach_state_49ers',
            'Cal Poly': 'cal_poly_mustangs',
            'Cal State Bakersfield': 'cal_state_bakersfield_roadrunners',
            'Cal State Fullerton': 'cal_state_fullerton_titans',
            'Campbell': 'campbell_fighting_camels',
            'Canisius': 'canisius_golden_griffs',
            'Central Arkansas': 'central_arkansas_bears',
            'Central Connecticut': 'central_connecticut_blue_devils',
            'Central Michigan': 'central_michigan_chippewas',
            'Charleston Southern': 'charleston_southern_buccaneers',
            'Charlotte': 'charlotte_49ers',
            'Chattanooga': 'chattanooga_mocs',
            'Cincinnati': 'cincinnati_bearcats',
            'Citadel': 'citadel_bulldogs',
            'Clemson': 'clemson_tigers',
            'Coastal Carolina': 'coastal_carolina_chanticleers',
            'Colgate': 'colgate_raiders',
            'College of Charleston': 'college_of_charleston_cougars',
            'Colonial Athletic Association': 'colonial_athletic_association',
            'Colorado': 'colorado_buffaloes',
            'Colorado State': 'colorado_state_rams',
            'Conference USA': 'conference_usa',
            'Connecticut': 'connecticut_huskies',
            'Coppin State': 'coppin_state_eagles',
            'Creighton': 'creighton_bluejays',
            'Cal State Northridge': 'csun_matadors',
            'Davidson': 'davidson_wildcats',
            'Dayton': 'dayton_flyers',
            'Delaware': 'delaware_fightin_blue_hens',
            'Delaware State': 'delaware_state_hornets',
            'Denver': 'denver_pioneers',
            'DePaul': 'depaul_blue_demons',
            'Drake': 'drake_bulldogs',
            'Drexel': 'drexel_dragons',
            'Duke': 'duke_blue_devils',
            'Duquesne': 'duquesne_dukes',
            'Eastern Illinois': 'eastern_illinois_panthers',
            'Eastern Kentucky': 'eastern_kentucky_colonels',
            'Eastern Michigan': 'eastern_michigan_eagles',
            'Eastern Washington': 'eastern_washington_eagles',
            'East Carolina': 'east_carolina_pirates',
            'ETSU': 'east_tennessee_state_buccaneers',
            'Elon': 'elon_phoenix',
            'Evansville': 'evansville_purple_aces',
            'Fairfield': 'fairfield_stags',
            'FDU': 'fairleigh_dickinson_fdu_knights',
            'FIU': 'fiu_panthers',
            'Florida A&M': 'florida_am_rattlers',
            'Florida Atlantic': 'florida_atlantic_owls',
            'Florida': 'florida_gators',
            'Florida Gulf Coast': 'florida_gulf_coast_eagles',
            'Florida State': 'florida_state_seminoles',
            'Fordham': 'fordham_rams',
            'Fresno State': 'fresno_state_bulldogs',
            'Furman': 'furman_paladins',
            'Gardner-Webb': 'gardner_webb_runnin_bulldogs',
            'Georgetown': 'georgetown_hoyas',
            'George Mason': 'george_mason_patriots',
            'George Washington': 'george_washington_colonials',
            'Georgia': 'georgia_bulldogs',
            'Georgia Southern': 'georgia_southern_eagles',
            'Georgia State': 'georgia_state_panthers',
            'Georgia Tech': 'georgia_tech_yellow_jackets',
            'Gonzaga': 'gonzaga_bulldogs',
            'Grambling': 'grambling_state_tigers',
            'Grand Canyon': 'grand_canyon_antelopes',
            'Hampton': 'hampton_pirates',
            'Hawaii': 'hawaii_rainbow_warriors',
            'High Point': 'high_point_panthers',
            'Hofstra': 'hofstra_pride',
            'Holy Cross': 'holy_cross_crusaders',
            'Houston Baptist': 'houston_baptist_huskies',
            'Houston': 'houston_cougars',
            'Howard': 'howard_bison',
            'Idaho State': 'idaho_state_bengals',
            'Idaho': 'idaho_vandals',
            'UIC': 'illinois_chicago_uic_flames',
            'Illinois': 'illinois_fighting_illini',
            'Illinois State': 'illinois_state_redbirds',
            'Incarnate Word': 'incarnate_word_cardinals',
            'Indiana': 'indiana_hoosiers',
            'Indiana State': 'indiana_state_sycamores',
            'Iona': 'iona_gaels',
            'Iowa': 'iowa_hawkeyes',
            'Iowa State': 'iowa_state_cyclones',
            'Jacksonville': 'jacksonville_dolphins',
            'Jacksonville State': 'jacksonville_state_gamecocks',
            'Jackson State': 'jackson_state_tigers',
            'James Madison': 'james_madison_dukes',
            'Kansas City': 'kansas_city_umkc_roos',
            'Kansas': 'kansas_jayhawks',
            'Kansas State': 'kansas_state_wildcats',
            'Kennesaw State': 'kennesaw_state_owls',
            'Kentucky': 'kentucky_wildcats',
            'Kent State': 'kent_state_golden_flashes',
            'Lafayette': 'lafayette_leopards',
            'Lamar': 'lamar_cardinals',
            'La Salle': 'la_salle_explorers',
            'Lehigh': 'lehigh_mountain_hawks',
            'Le Moyne': 'le_moyne_dolphins',
            'Liberty': 'liberty_flames',
            'Lindenwood': 'lindenwood_lions',
            'Lipscomb': 'lipscomb_bisons',
            'Longwood': 'longwood_lancers',
            'LIU': 'long_island_liu_sharks',
            'Louisiana': 'louisiana_lafayette_ragin_cajuns',
            'Louisiana Monroe': 'louisiana_monroe_warhawks',
            'Louisiana Tech': 'louisiana_tech_bulldogs',
            'Louisville': 'louisville_cardinals',
            'Loyola (IL)': 'loyola_chicago_ramblers',
            'Loyola Marymount': 'loyola_marymount_lions',
            'Loyola (MD)': 'loyola_university_maryland_greyhounds',
            'LSU': 'lsu_tigers',
            'Maine': 'maine_black_bears',
            'Manhattan': 'manhattan_jaspers',
            'Marist': 'marist_red_foxes',
            'Marquette': 'marquette_golden_eagles',
            'Marshall': 'marshall_thundering_herd',
            'Maryland Eastern Shore': 'maryland_eastern_shore_hawks',
            'Maryland': 'maryland_terrapins',
            'McNeese State': 'mcneese_state_cowboys',
            'Memphis': 'memphis_tigers',
            'Mercer': 'mercer_bears',
            'Merrimack': 'merrimack_warriors',
            'Metro Atlantic Athletic Conference':
                'metro_atlantic_athletic_conference_maac',
            'Miami': 'miami_hurricanes',
            'Miami OH': 'miami_oh_redhawks',
            'Michigan State': 'michigan_state_spartans',
            'Michigan': 'michigan_wolverines',
            'Middle Tennessee': 'middle_tennessee_blue_raiders',
            'Mid American Conference': 'mid_american_conference',
            'Mid Eastern Athletic Conference': 'mid_eastern_athletic_conference_meac',
            'Minnesota': 'minnesota_golden_gophers',
            'Mississippi State': 'mississippi_state_bulldogs',
            'Mississippi Valley State': 'mississippi_valley_state_delta_devils',
            'Missouri State': 'missouri_state_bears',
            'Missouri': 'missouri_tigers',
            'Missouri Valley Conference': 'missouri_valley_conference',
            'Monmouth': 'monmouth_hawks',
            'Montana': 'montana_grizzlies',
            'Montana State': 'montana_state_bobcats',
            'Morehead State': 'morehead_state_eagles',
            'Morgan State': 'morgan_state_bears',
            'Mountain West Conference': 'mountain_west_conference',
            "Mount St. Mary's": "mount_st._marys_mountaineers",
            'Murray State': 'murray_state_racers',
            'Navy': 'navy_midshipmen',
            'Nebraska': 'nebraska_cornhuskers',
            'Nebraska Omaha': 'nebraska_omaha_mavericks',
            'Nevada': 'nevada_wolf_pack',
            'New Hampshire': 'new_hampshire_wildcats',
            'New Jersey Institute Of Technology':
                'new_jersey_institute_of_technology_njit_highlanders',
            'New Mexico': 'new_mexico_lobos',
            'New Mexico State': 'new_mexico_state_aggies',
            'New Orleans': 'new_orleans_privateers',
            'Niagara': 'niagara_purple_eagles',
            'Nicholls State': 'norfolk_state_spartans',
            'Northeastern': 'northeastern_huskies',
            'Northeast Conference': 'northeast_conference',
            'Northern Arizona': 'northern_arizona_lumberjacks',
            'Northern Colorado': 'northern_colorado_bears',
            'Northern Illinois': 'northern_illinois_huskies_300x300',
            'Northern Iowa': 'northern_iowa_panthers',
            'Northwestern State': 'northwestern_state_demons',
            'Northwestern': 'northwestern_wildcats',
            'North Alabama': 'north_alabama_lions',
            'North Carolina A&T': 'north_carolina_at_aggies',
            'North Carolina Central': 'north_carolina_central_eagles',
            'NC State': 'north_carolina_state_wolfpack',
            'UNC': 'north_carolina_tar_heels',
            'North Dakota': 'north_dakota_fighting_hawks',
            'North Dakota State': 'north_dakota_state_bison',
            'North Florida': 'north_florida_ospreys',
            'North Texas': 'north_texas_mean_green',
            'Notre Dame': 'notre_dame_fighting_irish',
            'Ohio': 'ohio_bobcats',
            'Ohio State': 'ohio_state_buckeyes',
            'Ohio Valley Conference': 'ohio_valley_conference',
            'Oklahoma': 'oklahoma_sooners',
            'Oklahoma State': 'oklahoma_state_cowboys',
            'Old Dominion': 'old_dominion_monarchs',
            'Ole Miss': 'ole_miss_rebels',
            'Oral Roberts': 'oral_roberts_golden_eagles',
            'Oregon': 'oregon_ducks',
            'Oregon State': 'oregon_state_beavers',
            'Pacific': 'pacific_tigers',
            'Pac 12': 'pac_12',
            'Patriot League Conference': 'patriot_league_conference',
            'Penn State': 'penn_state_nittany_lions',
            'Penn': 'penn_quakers',
            'Pepperdine': 'pepperdine_waves',
            'Pitt': 'pitt_panthers',
            'Portland': 'portland_pilots',
            'Portland State': 'portland_state_vikings',
            'Prairie View AM': 'prairie_view_am_panthers',
            'Presbyterian': 'presbyterian_blue_hose',
            'Providence': 'providence_friars',
            'Purdue': 'purdue_boilermakers',
            'Queens University Of Charlotte': 'queens_university_of_charlotte_royals',
            'Quinnipiac': 'quinnipiac_bobcats',
            'Radford': 'radford_highlanders',
            'Rhode Island': 'rhode_island_rams',
            'Rice': 'rice_owls',
            'Richmond': 'richmond_spiders',
            'Rider': 'rider_broncs',
            'Rutgers': 'rutgers_scarlet_knights',
            'Sacramento State': 'sacramento_state_hornets',
            'Sacred Heart': 'sacred_heart_pioneers',
            'Saint Francis PA': 'saint_francis_pa_red_flash',
            "St. Joseph's": 'saint_josephs_hawks',
            'Saint Louis': 'saint_louis_billikens',
            "Saint Mary's": 'saint_marys_college_gaels',
            "St. Peter's": 'saint_peters_peacocks',
            'Samford': 'samford_bulldogs',
            'Sam Houston State': 'sam_houston_state_bearkats',
            'Santa Clara': 'santa_clara_broncos',
            'San Diego State': 'san_diego_state_aztecs',
            'San Diego': 'san_diego_toreros',
            'San Francisco': 'san_francisco_dons',
            'San Jose State': 'san_jose_state_spartans',
            'Seattle': 'seattle_redhawks',
            'Seton Hall': 'seton_hall_pirates',
            'Siena': 'siena_saints',
            'SMU': 'smu_mustang',
            'Southeastern Conference': 'southeastern_conference',
            'Southeastern Louisiana': 'southeastern_louisiana_lions',
            'Southeast Missouri State': 'southeast_missouri_state_redhawks',
            'Southern Conference': 'southern_conference',
            'Southern Illinois': 'southern_illinois_salukis',
            'Southern Illinois University Edwardsville':
                'southern_illinois_university_edwardsville_siue_cougars',
            'Southern Indiana': 'southern_indiana_screaming_eagles',
            'Southern': 'southern_jaguars',
            'Southern Miss': 'southern_miss_golden_eagles',
            'Southern Utah': 'southern_utah_thunderbirds',
            'Southland Conference': 'southland_conference',
            'Southwestern Athletic Conference': 'southwestern_athletic_conference',
            'South Alabama': 'south_alabama_jaguars',
            'South Carolina': 'south_carolina_gamecocks',
            'South Carolina State': 'south_carolina_state_bulldogs',
            'South Carolina Upstate': 'south_carolina_upstate_spartans',
            'South Dakota': 'south_dakota_coyotes',
            'South Dakota State': 'south_dakota_state_jackrabbits',
            'South Florida': 'south_florida_bulls',
            'St. Bonaventure': 'st._bonaventure_bonnies',
            'St. Francis Brooklyn': 'st._francis_brooklyn_terriers',
            "St. John's": 'st._johns_red_storm',
            'St. Thomas': 'st._thomas_tommies',
            'Stanford': 'stanford_cardinal',
            'Stephen F. Austin': 'stephen_f._austin_lumberjacks',
            'Stetson': 'stetson_hatters',
            'Stonehill': 'stonehill_skyhawks',
            'Stony Brook': 'stony_brook_seawolves',
            'Summit League': 'summit_league',
            'Sun Belt Conference 2020': 'sun_belt_conference_2020',
            'Syracuse': 'syracuse_orange',
            'Tarleton State': 'tarleton_state_texans',
            'TCU': 'tcu_horned_frogs',
            'Temple': 'temple_owls',
            'Tennessee Martin': 'tennessee_martin_skyhawks',
            'Tennessee State': 'tennessee_state_tigers',
            'Tennessee Tech': 'tennessee_tech_golden_eagles',
            'Tennessee': 'tennessee_volunteers',
            'Texas A&M Commerce': 'texas_am_commerce_lions',
            'Texas A&M Corpus Christi': 'texas_am_corpus_christi_islanders',
            'Texas A&M': 'texas_am_university',
            'Texas': 'texas_longhorns',
            'Texas Rio Grande Valley': 'texas_rio_grande_valley_utrgv_vaqueros',
            'UTSA': 'texas_sa_roadrunners',
            'Texas Southern': 'texas_southern_tigers',
            'Texas State': 'texas_state_bobcats',
            'Texas Tech': 'texas_tech_red_raiders',
            'Toledo': 'toledo_rockets',
            'Towson': 'towson_tigers',
            'Troy': 'troy_trojans',
            'Tulane': 'tulane_green_wave',
            'Tulsa': 'tulsa_golden_hurricane',
            'UCF': 'ucf_knights',
            'UCLA': 'ucla_bruins',
            'UC-Davis': 'uc_davis_aggies',
            'UC-Irvine': 'uc_irvine_anteaters',
            'UC-Riverside': 'uc_riverside_highlanders',
            'UCSB': 'uc_santa_barbara_gauchos',
            'UC-San Diego': 'uc_san_diego_tritons',
            'UMass': 'umass_amherst_minutemen',
            'UMass Lowell': 'umass_lowell_river_hawks',
            'UNCG': 'uncg_spartans',
            'UNC Asheville': 'unc_asheville_bulldogs',
            'UNC Wilmington': 'unc_wilmington_seahawks',
            'UMBC': 'university_of_maryland_baltimore_county_umbc_retrievers',
            'UNLV': 'unlv_rebels',
            'USC': 'usc_trojans',
            'Utah State': 'utah_state_aggies',
            'Utah Tech': 'utah_tech_trailblazers',
            'Utah': 'utah_utes',
            'Utah Valley': 'utah_valley_wolverines',
            'UTEP': 'utep_miners',
            'UT Arlington': 'ut_arlington_mavericks',
            'Valparaiso': 'valparaiso_beacons',
            'Vanderbilt': 'vanderbilt_commodores',
            'VCU': 'vcu_rams',
            'Vermont': 'vermont_catamounts',
            'Villanova': 'villanova_wildcats',
            'Virginia': 'virginia_cavaliers',
            'Virginia Tech': 'virginia_tech_hokies',
            'VMI': 'vmi_keydets',
            'Wagner': 'wagner_seahawks',
            'Wake Forest': 'wake_forest_demon_deacons',
            'Washington': 'washington_huskies',
            'Washington State': 'washington_state_cougars',
            'Weber State': 'weber_state_wildcats',
            'Western Athletic Conference': 'western_athletic_conference',
            'Western Carolina': 'western_carolina_catamounts',
            'Western Illinois': 'western_illinois_leathernecks',
            'Western Kentucky': 'western_kentucky_hilltoppers',
            'Western Michigan': 'western_michigan_broncos',
            'West Coast Conference': 'west_coast_conference',
            'West Virginia': 'west_virginia_mountaineers',
            'Wichita State': 'wichita_state_shockers',
            'William Mary': 'william_mary_tribe',
            'Winthrop': 'winthrop_eagles',
            'Wisconsin': 'wisconsin_badgers',
            'Wofford': 'wofford_terriers',
            'Wyoming': 'wyoming_cowboys',
            'Xavier': 'xavier_musketeers',
            'Yale': 'yale_bulldogs',
            'Brown': 'brown_bears',
            'Cleveland State': 'cleveland_state_vikings',
            'Columbia': 'columbia_lions',
            'Cornell': 'cornell_big_red',
            'Dartmouth': 'dartmouth_big_green',
            'Detroit Mercy': 'detroit_mercer_hawks',
            'Harvard': 'harvard_crimson',
            'IUPUI': 'iupui_jaguars',
            'Northern Kentucky': 'northern_kentucky_norse',
            'Oakland': 'oakland_michigan_golden_grizzlies',
            'Princeton': 'princeton_tigers',
            'Purdue Fort Wayne': 'purdue_fort_wayne_mastodons',
            'Robert Morris': 'robert_morris_colonials',
            'Green Bay': 'wisconsin_green_bay_phoenix',
            'Milwaukee': 'wisconsin_milwaukee_panthers',
            'Wright State': 'wright_state_raiders',
            'Youngstown State': 'youngstown_state_penguins',
            'Norfolk State': 'norfolk_state_spartans',
            # Add more mappings as needed, similar to your Flutter code
        }
        
        # Look for exact matches first
        if team_name in team_name_map:
            return team_name_map[team_name]
        
        # Try to find partial matches
        for key, value in team_name_map.items():
            if key in team_name:
                return value
        
        # If no match found, create a normalized version of the name
        normalized = team_name.lower()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        normalized = normalized.replace(' ', '_')
        
        return normalized
    
    def predict_winner(self):
        #team1, team2, round, year1, year2
        print(self.left_team, self.right_team, self.selected_round, self.left_year, self.right_year)
        
        team1_data = get_data(int(self.left_year))
        team1_data = team1_data[team1_data['school'] == self.left_team]

        team2_data = get_data(int(self.right_year))
        team2_data = team2_data[team2_data['school'] == self.right_team]

        processed_data = preprocess_data(team1_data, team2_data, self.selected_round)

        prediction_proba = self.lgbm_model.predict_proba(processed_data)[0]

        team1_win_prob = round(prediction_proba[1] * 100, 2)
        team2_win_prob = round(prediction_proba[0] * 100, 2)

        print(team1_win_prob, team2_win_prob)
        
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NCAATeamMatchupApp()
    window.show()
    sys.exit(app.exec_())