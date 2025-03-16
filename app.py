import sys
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QPushButton, QFrame, QGridLayout, QSizePolicy,
                            QMessageBox)
from PyQt5.QtGui import QPixmap, QColor, QLinearGradient, QPainter, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QSize, QRect, QTimer, QUrl, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtNetwork import QNetworkRequest, QNetworkAccessManager, QNetworkReply
import joblib
import pandas as pd
info = pd.read_csv('data/game_results/games_2025.csv')

from helper import get_data, merge_data

def get_team_color(team):
    team_data = info[info['team_location'] == team]
    hex_color = str(team_data['team_color'].iloc[0])
    if is_dark_color(hex_color):
        hex_color = str(team_data['team_alternate_color'].iloc[0])
    return "#" + hex_color, "#FFFFFF"

def is_dark_color(hex_color, threshold=30):
    """
    Check if a hex color is black or very dark
    
    Parameters:
    -----------
    hex_color : str
        Hex color code (with or without # prefix)
    threshold : int
        RGB value below which a color is considered dark (0-255)
        
    Returns:
    --------
    bool
        True if the color is dark, False otherwise
    """
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        # Invalid hex color
        return False
    
    # Check if all RGB values are below threshold
    return all(value < threshold for value in (r, g, b))

def get_logo_url(team_name):
    logo_url = info[info['team_location'] == team_name]
    url = logo_url['team_logo'].iloc[0]        
    return url

# Default team colors for teams not in the dictionary
DEFAULT_TEAM_COLOR = ('#333333', '#FFFFFF')  # Dark gray and white

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
        self.team_colors = DEFAULT_TEAM_COLOR
        
        # Set side colors
        self.side_colors = (QColor("#2E2C2B"), QColor("#000000")) if is_left else (QColor("#6A6260"), QColor("#86807F"))
        self.win_gradient = False
        self.win_probability = 0.0
        
        # Border animation properties
        self.border_opacity = 0.0
        self.border_animation_timer = QTimer(self)
        self.border_animation_timer.timeout.connect(self.animate_border)
        self.border_animation_direction = 1  # 1 for increasing, -1 for decreasing
        
        # Main layout with more spacing for centered appearance
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 30, 20, 30)  # Add more margin around elements
        self.layout.setSpacing(20)  # Increase spacing between elements
        self.setLayout(self.layout)
        
        # Rest of your initialization code...
        # Top section layout
        top_layout = QHBoxLayout()
        
        # Conference logo - moved to outer corner and MUCH larger
        self.conf_logo_label = QLabel()
        self.conf_logo_label.setAlignment(Qt.AlignCenter)
        self.conf_logo_label.setFixedSize(120, 120)  # Much larger conference logo
        self.conf_logo_label.setStyleSheet("background-color: transparent; color: white; font-size: 14px;")
        
        # Year selector - moved to inner corner
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
        
        # Position based on side (left or right)
        if self.is_left:
            # For left side: conf logo on left, year on right
            top_layout.addWidget(self.conf_logo_label)
            top_layout.addStretch(1)
            top_layout.addLayout(year_layout)
        else:
            # For right side: year on left, conf logo on right
            top_layout.addLayout(year_layout)
            top_layout.addStretch(1) 
            top_layout.addWidget(self.conf_logo_label)
            
        self.layout.addLayout(top_layout)
        
        # Add stretch to push logo to center vertically
        self.layout.addStretch(1)
        
        # Team logo container - make it larger
        logo_container = QHBoxLayout()
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMinimumSize(400, 400)  # Increased size for larger logos
        self.logo_label.setStyleSheet("background-color: transparent;")
        
        logo_container.addStretch(1)
        logo_container.addWidget(self.logo_label)
        logo_container.addStretch(1)
        
        self.layout.addLayout(logo_container)
        
        # Win probability label with enhanced style for better visibility
        self.probability_label = QLabel("")
        self.probability_label.setAlignment(Qt.AlignCenter)
        self.probability_label.setStyleSheet("""
            color: white; 
            font-size: 18px; 
            font-weight: bold; 
            background-color: transparent;
            padding: 10px;
        """)
        
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

    def animate_border(self):
        # Update border opacity based on direction
        self.border_opacity += 0.05 * self.border_animation_direction
        
        # Reverse direction at limits
        if self.border_opacity >= 1.0:
            self.border_opacity = 1.0
            self.border_animation_direction = -1
        elif self.border_opacity <= 0.3:
            self.border_opacity = 0.3
            self.border_animation_direction = 1
            
        # Update the widget to repaint with new opacity
        self.update()
    
    def set_win_gradient(self, is_winner, probability):
        self.win_gradient = is_winner
        self.win_probability = probability
        
        # Set probability text with enhanced style for winner
        if is_winner:
            self.probability_label.setText(f"Win Probability: {probability:.1f}%")
            self.probability_label.setStyleSheet("""
                color: gold; 
                font-size: 24px; 
                font-weight: bold; 
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 12px;
                padding: 10px;
            """)
            
            # Start border animation for winner
            if not self.border_animation_timer.isActive():
                self.border_opacity = 0.3  # Starting opacity
                self.border_animation_timer.start(50)  # 50ms interval for smooth animation
        else:
            self.probability_label.setText(f"Win Probability: {probability:.1f}%")
            self.probability_label.setStyleSheet("""
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                background-color: transparent;
                padding: 10px;
            """)
            
            # Stop border animation for non-winner
            if self.border_animation_timer.isActive():
                self.border_animation_timer.stop()
        
        # Update the background
        self.update()
    
    def paintEvent(self, event):
        # Create painter
        painter = QPainter(self)
        
        # Create gradient background
        gradient = QLinearGradient()
        
        # Get team primary and secondary colors
        primary_color = QColor(self.team_colors[0])
        secondary_color = QColor(self.team_colors[1])
        
        # Adjust opacity to make colors less intense
        primary_color.setAlpha(180)
        secondary_color.setAlpha(150)
        
        # Set gradient direction based on side
        if self.is_left:
            gradient.setStart(self.width(), self.height() / 2)
            gradient.setFinalStop(0, self.height() / 2)
        else:
            gradient.setStart(0, self.height() / 2)
            gradient.setFinalStop(self.width(), self.height() / 2)
        
        # Set gradient colors
        black_overlay = QColor("#000000")
        black_overlay.setAlpha(180)
        
        if self.is_left:
            gradient.setColorAt(0, primary_color)
            gradient.setColorAt(1, black_overlay)
        else:
            gradient.setColorAt(0, black_overlay)
            gradient.setColorAt(1, primary_color)
        
        # Fill background with gradient
        painter.fillRect(self.rect(), gradient)
        
        # Draw animated border for winner
        if self.win_gradient:
            # Create a golden border color with dynamic opacity
            border_color = QColor(255, 215, 0, int(255 * self.border_opacity))  # Gold color
            
            # Set pen for drawing border
            pen = QPen(border_color)
            pen.setWidth(6)  # Thicker border
            painter.setPen(pen)
            
            # Draw border around the widget with rounded corners
            painter.drawRoundedRect(3, 3, self.width() - 6, self.height() - 6, 15, 15)
            
            # Add a subtle glow effect
            glow_pen = QPen(QColor(255, 215, 0, int(50 * self.border_opacity)))
            glow_pen.setWidth(10)
            painter.setPen(glow_pen)
            painter.drawRoundedRect(5, 5, self.width() - 10, self.height() - 10, 15, 15)

    def update_conference_logo(self, conference):
        if not conference:
            self.conf_logo_label.clear()
            return
            
        # Format conference name for file path
        conf_name = conference.lower().replace(' ', '_')
        logo_path = os.path.join("assets", "logos", f"{conf_name}.png")
        
        # Create a circular background for the logo
        self.conf_logo_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.4); /* semi-transparent white */
            border: 2px solid rgba(255, 255, 255, 0.1); /* more visible white border */
            border-radius: 60px; /* half of width/height for perfect circle */
            padding: 10px;
        """)
        
        # Try to load the conference logo
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.conf_logo_label.setPixmap(pixmap)
        else:
            # Display text if logo not found
            self.conf_logo_label.setText(conference)
            self.conf_logo_label.setStyleSheet("""
                color: white; 
                font-size: 18px; 
                font-weight: bold; 
                background-color: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 60px;
                padding: 10px;
            """)
    
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
            
        # Update team colors
        self.update_team_colors(team)
        
        # Update team logo
        self.update_logo(team)
        
        # Update stats and UI
        self.parent_app.update_team_stats()
        
        # Reset win gradient when team changes
        self.win_gradient = False
        self.win_probability = 0.0
        self.probability_label.setText("")
        
        # Stop border animation if running
        if self.border_animation_timer.isActive():
            self.border_animation_timer.stop()
            
        self.update()
    
    def update_team_colors(self, team):
        self.team_colors = get_team_color(team)
        # Update the UI with new colors
        self.update()
    
    def update_logo(self, team):
        if not team:
            # If no team selected, show placeholder
            self.logo_label.setText("")
            self.logo_label.setStyleSheet("color: white; font-size: 180px; font-weight: bold; background-color: transparent;")
            return
        
        try:
            # Get normalized team name
            logo_url = get_logo_url(team)
            
            # Create network request
            self.network_manager = QNetworkAccessManager()
            request = QNetworkRequest(QUrl(logo_url))
            
            # Connect signal to handle the finished download
            self.network_manager.finished.connect(self.handle_logo_response)
            
            # Start the request
            self.reply = self.network_manager.get(request)
            
            # Store the team name to use in the callback
            self.current_team = team
            
            # Show a loading indicator while waiting
            self.logo_label.setText("Loading...")
            
        except Exception as e:
            print(f"Error loading logo for {team}: {str(e)}")
            # Use first letter as placeholder
            self.logo_label.setText(team[0] if team else "")
            self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
            
    @pyqtSlot(QNetworkReply)
    def handle_logo_response(self, reply):
        pixmap = QPixmap()
        
        if reply.error() == QNetworkReply.NoError:
            # Read image data and load into pixmap
            image_data = reply.readAll()
            pixmap.loadFromData(image_data)
            
            if not pixmap.isNull():
                # Scale to larger size for better visibility (400x400)
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pixmap)
                self.logo_label.setText("")
            else:
                # Fallback if image data couldn't be loaded
                self.logo_label.setText(self.current_team[0] if self.current_team else "")
                self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
        else:
            # Handle error
            print(f"Error downloading logo for {self.current_team}: {reply.errorString()}")
            # Use first letter as placeholder with larger font
            self.logo_label.setText(self.current_team[0] if self.current_team else "")
            self.logo_label.setStyleSheet(f"color: {self.team_colors[0] if hasattr(self, 'team_colors') and self.team_colors else 'white'}; font-size: 180px; font-weight: bold; background-color: transparent;")
        
        # Clean up
        reply.deleteLater()

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
        self.left_year = "2025"
        self.right_year = "2025"
        self.selected_round = "First Round"
        
        self.left_team_stats = TeamStats()
        self.right_team_stats = TeamStats()
        
        self.all_teams = []
        self.teams_by_year = {}
        self.all_team_stats = {}
        
        self.available_years = [
            "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", 
            "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008", "2007", "2006", 
            "2005", "2004", "2003"
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
        header.setFixedHeight(120)
        
        # Logo and title
        logo_label = QLabel()
        logo_pixmap = QPixmap("assets/logos/March_Madness_logo.png")
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(300, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        
        # NEW - Connect round changed signal
        self.round_combo.currentTextChanged.connect(self.round_changed)
        
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
    
    def round_changed(self, round_text):
        self.selected_round = round_text
        # Reset the prediction highlights
        self.left_team_widget.set_win_gradient(False, 0.0)
        self.right_team_widget.set_win_gradient(False, 0.0)
    
    def load_teams_from_csv(self):
        try:
            # Open and read the CSV file
            with open('assets/all_ratings.csv', 'r', encoding='utf-8') as file:
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
        # Check if selected years exist in data
        if self.left_year not in self.all_team_stats:
            print(f"WARNING: Year {self.left_year} not found in data")
        if self.right_year not in self.all_team_stats:
            print(f"WARNING: Year {self.right_year} not found in data")
        
        # Update left team stats
        if (self.left_team and self.left_year in self.all_team_stats and 
            self.left_team in self.all_team_stats[self.left_year]):
            self.left_team_stats = self.all_team_stats[self.left_year][self.left_team]
            # Update conference logo for left team
            self.left_team_widget.update_conference_logo(self.left_team_stats.conference)
        else:
            print(f"No stats found for {self.left_team} in {self.left_year} - using empty stats")
            self.left_team_stats = TeamStats()
            self.left_team_widget.conf_logo_label.clear()
        
        # Update right team stats
        if (self.right_team and self.right_year in self.all_team_stats and 
            self.right_team in self.all_team_stats[self.right_year]):
            self.right_team_stats = self.all_team_stats[self.right_year][self.right_team]
            # Update conference logo for right team
            self.right_team_widget.update_conference_logo(self.right_team_stats.conference)
        else:
            print(f"No stats found for {self.right_team} in {self.right_year} - using empty stats")
            self.right_team_stats = TeamStats()
            self.right_team_widget.conf_logo_label.clear()
        
        # Update stats comparison UI
        self.update_stats_comparison()
        
        # Reset any prediction highlights
        self.left_team_widget.set_win_gradient(False, 0.0)
        self.right_team_widget.set_win_gradient(False, 0.0)
    
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
    
    def predict_winner(self):
        # Validate that both teams have data
        if not self.left_team or not self.right_team:
            QMessageBox.warning(self, "Missing Data", "Please select both teams before predicting.")
            return
            
        print(f"Predicting: {self.left_team} ({self.left_year}) vs {self.right_team} ({self.right_year}) in {self.selected_round}")
        
        try:
            # Get team data for the selected years
            left_year_int = int(self.left_year)
            right_year_int = int(self.right_year)
            
            team1_data = get_data(self.left_team,left_year_int)
            team1_data.reset_index(inplace=True)
            team1_data['index'] = team1_data.index
            print(team1_data)

            team2_data = get_data(self.right_team,right_year_int)
            team2_data.reset_index(inplace=True)
            team2_data['index'] = team2_data.index
            print(team2_data)

            team1_data = team1_data.drop(columns=['Unnamed: 0'])
            team2_data = team2_data.drop(columns=['Unnamed: 0'])

            if team1_data.empty or team2_data.empty:
                QMessageBox.warning(self, "Missing Data", 
                                   f"Could not find data for {self.left_team} ({self.left_year}) or {self.right_team} ({self.right_year}).")
                return
                
            # Process data in both directions to reduce home/away bias
            processed_data = merge_data(team1_data, team2_data, left_year_int, right_year_int)
            processed_data = processed_data.drop(columns=['index','target','game_id_diff','game_date', 'home_team', 'home_color','away_team','away_color'])
            processed_data.fillna(0, inplace=True)

            inverted_data = merge_data(team2_data, team1_data, right_year_int, left_year_int)
            inverted_data = inverted_data.drop(columns=['index','target','game_id_diff','game_date', 'home_team', 'home_color','away_team','away_color'])
            inverted_data.fillna(0, inplace=True)

            ordered_data = pd.DataFrame()
            inverted_ordered_data = pd.DataFrame()

            columns = ['season_type', 'points_per_game_diff', 'assists_per_game_diff',
            'blocks_per_game_diff', 'defensive_rebounds_per_game_diff',
            'field_goal_pct_per_game_diff', 'field_goals_made_per_game_diff',
            'field_goals_attempted_per_game_diff', 'flagrant_fouls_per_game_diff',
            'fouls_per_game_diff', 'free_throw_pct_per_game_diff',
            'free_throws_made_per_game_diff', 'free_throws_attempted_per_game_diff',
            'offensive_rebounds_per_game_diff', 'steals_per_game_diff',
            'team_turnovers_per_game_diff', 'technical_fouls_per_game_diff',
            'three_point_field_goal_pct_per_game_diff',
            'three_point_field_goals_made_per_game_diff',
            'three_point_field_goals_attempted_per_game_diff',
            'total_rebounds_per_game_diff', 'total_technical_fouls_per_game_diff',
            'total_turnovers_per_game_diff', 'turnovers_per_game_diff',
            'opponent_points_per_game_diff', 'largest_lead_per_game_diff',
            'wins_diff', 'losses_diff', 'win_loss_percentage_diff','momentum_diff','SOS_diff',
            'fast_break_points_per_game_diff', 'points_in_paint_per_game_diff',
            'turnover_points_per_game_diff']

            for column in columns:
                try:
                    ordered_data[column] = processed_data[column]
                except:
                    ordered_data[column] = 0

            for column in columns:
                try:
                    inverted_ordered_data[column] = inverted_data[column]
                except:
                    inverted_ordered_data[column] = 0

            
            # Get predictions
            proba_1_lgbm = self.lgbm_model.predict_proba(ordered_data)[0]
            proba_2_lgbm = self.lgbm_model.predict_proba(inverted_ordered_data)[0]

            print(proba_1_lgbm)
            print(proba_2_lgbm)

            # Direction 1: team1 vs team2
            team1_proba_dir1 = proba_1_lgbm[1]

            # Direction 2: team2 vs team1 (need to invert to get team1's perspective)
            team1_proba_dir2 = proba_2_lgbm[0]

            # Average the two directions
            team1_proba = (team1_proba_dir1 + team1_proba_dir2) / 2
            team2_proba = 1 - team1_proba  # These should sum to 1
            
            # Convert to percentages
            team1_win_prob = round(team1_proba * 100, 2)
            team2_win_prob = round(team2_proba * 100, 2)

            # Random factor for close games
            if abs(team1_win_prob - team2_win_prob) < 15:
                print("Close game! Adding random factor to prediction.")
                import random
                random_value = random.uniform(0, 1)
                team1_wins = random_value < (team1_win_prob/100)
                print(f"Random value: {random_value}, Team1 prob: {team1_win_prob}")
            else:
                # Determine the winner based on average probability
                team1_wins = team1_win_prob > team2_win_prob
            
            # Update UI to highlight winner and show probabilities
            self.left_team_widget.set_win_gradient(team1_wins, team1_win_prob)
            self.right_team_widget.set_win_gradient(not team1_wins, team2_win_prob)
            
            # Show a message with the result
            winner = self.left_team if team1_wins else self.right_team
            winner_prob = team1_win_prob if team1_wins else team2_win_prob
            
            prediction_text = f"Predicted Winner: {winner} ({winner_prob:.1f}% chance)\n\n"
            prediction_text += f"{self.left_team}: {team1_win_prob:.1f}% chance\n"
            prediction_text += f"{self.right_team}: {team2_win_prob:.1f}% chance"
            
            if abs(team1_win_prob - team2_win_prob) < 15:
                prediction_text += "\n\n(Close matchup! Added random factor to prediction)"
                
            QMessageBox.information(self, "Prediction Result", prediction_text)
            
        except Exception as e:
            print(f"Error predicting winner: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {str(e)}")
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NCAATeamMatchupApp()
    window.show()
    sys.exit(app.exec_())