import 'package:flutter/material.dart';
import 'dart:math' show max;
import 'package:csv/csv.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ESPN-Style Bracket',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const BracketScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class BracketScreen extends StatefulWidget {
  const BracketScreen({super.key});

  @override
  State<BracketScreen> createState() => _BracketScreenState();
}

class _BracketScreenState extends State<BracketScreen> {
  String tournamentYear = "2000"; // Default year
  String currentView = "full";
  Map<String, List<Map<String, dynamic>>> _matchesByRegion = {};
  bool _isLoading = true;

  List<String> availableYears = [
    "2025",
    "2024",
    "2023",
    "2022",
    "2021",
    "2020",
    "2019",
    "2018",
    "2017",
    "2016",
    "2015",
    "2014",
    "2013",
    "2012",
    "2011",
    "2010",
    "2009",
    "2008",
    "2007",
    "2006",
    "2005",
    "2004",
    "2003",
    "2002",
    "2001",
    "2000",
    "1999",
    "1998",
    "1997",
    "1996",
    "1995",
    "1994",
    "1993",
    "1992",
    "1991",
    "1990",
    "1989",
    "1988",
    "1987",
    "1986",
    "1985",
  ]; // Add more years as needed

  @override
  void initState() {
    super.initState();
    _loadMatchesFromCsv();
  }

  // Load and parse CSV file
  Future<void> _loadMatchesFromCsv() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final String csvData = await DefaultAssetBundle.of(
        context,
      ).loadString('assets/all_tournaments.csv');
      final List<List<dynamic>> csvTable = const CsvToListConverter().convert(
        csvData,
      );

      final tempMatchesByRegion = <String, List<Map<String, dynamic>>>{};

      for (var i = 1; i < csvTable.length; i++) {
        final row = csvTable[i];

        if (row.length < 8 || row[9].toString() != tournamentYear) {
          continue;
        }

        final region = row[0].toString().toLowerCase();
        final match = {
          'teamA': row[2].toString(),
          'teamB': row[5].toString(),
          'scoreA': row[4].toString(),
          'scoreB': row[7].toString(),
          'seedA': row[3].toString(),
          'seedB': row[6].toString(),
          'round': row[1].toString(),
          'csvIndex': i,
        };

        tempMatchesByRegion.putIfAbsent(region, () => []).add(match);
      }

      setState(() {
        _matchesByRegion = tempMatchesByRegion;
        _isLoading = false;
      });
    } catch (e) {
      print('Error loading CSV: $e');
    }
  }

  // Then update your getSampleMatches method to preserve CSV order
  List<Map<String, dynamic>> getSampleMatches(String region) {
    // If matches are still loading or region doesn't exist, return empty list
    if (_isLoading || !_matchesByRegion.containsKey(region)) {
      return [];
    }

    // Get all matches for this region
    final List<Map<String, dynamic>> regionMatches = List.from(
      _matchesByRegion[region]!,
    );

    // Sort matches by original CSV index to preserve order
    regionMatches.sort((a, b) {
      final indexA = a['csvIndex'] as int? ?? 0;
      final indexB = b['csvIndex'] as int? ?? 0;
      return indexA.compareTo(indexB);
    });

    return regionMatches;
  }

  // In your build method, add a loading indicator
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: Row(
          children: [
            const Icon(Icons.sports_basketball, color: Colors.white),
            const SizedBox(width: 8),
            const Text(
              'NCAA Tournament Bracket',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Spacer(),
            // Dropdown menu for selecting the tournament year
            DropdownButton<String>(
              value: tournamentYear,
              dropdownColor: Colors.black,
              style: const TextStyle(color: Colors.white),
              icon: const Icon(Icons.arrow_drop_down, color: Colors.white),
              onChanged: (String? newValue) {
                if (newValue != null) {
                  setState(() {
                    tournamentYear = newValue;
                    _loadMatchesFromCsv(); // Reload data for selected year
                  });
                }
              },
              items:
                  availableYears.map<DropdownMenuItem<String>>((String year) {
                    return DropdownMenuItem<String>(
                      value: year,
                      child: Text(
                        year,
                        style: const TextStyle(color: Colors.white),
                      ),
                    );
                  }).toList(),
            ),
          ],
        ),
      ),
      body:
          _isLoading
              ? const Center(child: CircularProgressIndicator())
              : Column(
                children: [
                  buildRegionSelector(),
                  Expanded(
                    child:
                        currentView == "full"
                            ? buildFullBracket()
                            : buildRegionalBracket(currentView),
                  ),
                  buildBottomInfoBar(),
                ],
              ),
    );
  }

  Widget buildRegionSelector() {
    return Container(
      color: Colors.grey.shade200,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            buildRegionButton("FULL BRACKET", "full"),
            const SizedBox(width: 8),
            buildRegionButton("EAST", "east"),
            const SizedBox(width: 8),
            buildRegionButton("WEST", "west"),
            const SizedBox(width: 8),
            buildRegionButton("SOUTH", "south"),
            const SizedBox(width: 8),
            buildRegionButton("MIDWEST", "midwest"),
          ],
        ),
      ),
    );
  }

  Widget buildRegionButton(String label, String region) {
    final isSelected = currentView == region;

    return InkWell(
      onTap: () {
        setState(() {
          currentView = region;
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? Colors.red : Colors.transparent,
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.black,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            fontSize: 13,
          ),
        ),
      ),
    );
  }

  Widget buildFullBracket() {
    return Center(
      child: Text(
        "Full Bracket View\n(Horizontal scrolling with all regions)",
        textAlign: TextAlign.center,
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.bold,
          color: Colors.grey.shade700,
        ),
      ),
    );
  }

  Widget buildRegionalBracket(String region) {
    // Get all matches for the region, preserving CSV order
    final allMatches = getSampleMatches(region);

    // Filter matches by round, preserving original order
    final round1Matches =
        allMatches
            .where(
              (match) =>
                  match['round'].toString().toLowerCase().contains('first') ||
                  match['round'] == '1',
            )
            .toList();

    final round2Matches =
        allMatches
            .where(
              (match) =>
                  match['round'].toString().toLowerCase().contains('second') ||
                  match['round'] == '2',
            )
            .toList();

    final round3Matches =
        allMatches
            .where(
              (match) =>
                  match['round'].toString().toLowerCase().contains('sweet') ||
                  match['round'] == '3',
            )
            .toList();

    final round4Matches =
        allMatches
            .where(
              (match) =>
                  match['round'].toString().toLowerCase().contains('elite') ||
                  match['round'] == '4',
            )
            .toList();

    // Set consistent spacing values
    const double matchCardHeight = 85.0; // Height for match cards + spacing
    const double matchCardWidth = 180.0; // Fixed width for match cards
    const double horizontalSpacing = 50.0; // Space between rounds

    // Dynamically calculate vertical offsets based on actual number of matches
    List<double> round1Offsets = [];
    for (int i = 0; i < round1Matches.length; i++) {
      round1Offsets.add(i * matchCardHeight);
    }

    // Calculate positions for second round matches
    List<double> round2Offsets = [];
    for (int i = 0; i < round2Matches.length; i++) {
      // Center each match between its corresponding first round matches
      double offset = i * matchCardHeight * 2 + matchCardHeight / 2;
      round2Offsets.add(offset);
    }

    // Calculate positions for Sweet 16 matches
    List<double> round3Offsets = [];
    for (int i = 0; i < round3Matches.length; i++) {
      // Center each match between its corresponding second round matches
      double offset = i * matchCardHeight * 4 + matchCardHeight * 1.5;
      round3Offsets.add(offset);
    }

    // Calculate position for Elite 8 match
    double round4Offset = matchCardHeight * 3.5; // Center vertically

    // Calculate total height needed for the bracket
    double totalHeight = max(
      round1Matches.length * matchCardHeight + 80,
      720.0, // Minimum height
    );

    return SingleChildScrollView(
      child: Column(
        children: [
          buildRegionHeader(region.toUpperCase()),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Container(
              padding: const EdgeInsets.all(16.0),
              width: 900, // Total width for all rounds + spacing
              height: totalHeight, // Dynamic height based on number of matches
              child: Stack(
                children: [
                  // Round labels at the top
                  Positioned(
                    left: 0,
                    top: 0,
                    child: SizedBox(
                      width: 900,
                      child: Row(
                        children: [
                          SizedBox(
                            width: matchCardWidth,
                            child: buildRoundLabel("1st Round", "Mar 21-22"),
                          ),
                          SizedBox(width: horizontalSpacing),
                          SizedBox(
                            width: matchCardWidth,
                            child: buildRoundLabel("2nd Round", "Mar 23-24"),
                          ),
                          SizedBox(width: horizontalSpacing),
                          SizedBox(
                            width: matchCardWidth,
                            child: buildRoundLabel("Sweet 16", "Mar 28-29"),
                          ),
                          SizedBox(width: horizontalSpacing),
                          SizedBox(
                            width: matchCardWidth,
                            child: buildRoundLabel("Elite 8", "Mar 30-31"),
                          ),
                        ],
                      ),
                    ),
                  ),

                  // Round 1 matches
                  ...List.generate(round1Matches.length, (index) {
                    return Positioned(
                      left: 0,
                      top:
                          40 +
                          (index < round1Offsets.length
                              ? round1Offsets[index]
                              : 0),
                      child: buildMatchCard(round1Matches[index], false),
                    );
                  }),

                  // Round 2 matches
                  ...List.generate(round2Matches.length, (index) {
                    return Positioned(
                      left: matchCardWidth + horizontalSpacing,
                      top:
                          40 +
                          (index < round2Offsets.length
                              ? round2Offsets[index]
                              : 0),
                      child: buildMatchCard(round2Matches[index], false),
                    );
                  }),

                  // Sweet 16 matches
                  ...List.generate(round3Matches.length, (index) {
                    return Positioned(
                      left: 2 * (matchCardWidth + horizontalSpacing),
                      top:
                          40 +
                          (index < round3Offsets.length
                              ? round3Offsets[index]
                              : 0),
                      child: buildMatchCard(round3Matches[index], false),
                    );
                  }),

                  // Elite 8 match
                  if (round4Matches.isNotEmpty)
                    Positioned(
                      left: 3 * (matchCardWidth + horizontalSpacing),
                      top: 40 + round4Offset,
                      child: buildMatchCard(round4Matches[0], true),
                    ),

                  // Connection lines for Round 1 to Round 2
                  if (round1Matches.isNotEmpty && round2Matches.isNotEmpty)
                    CustomPaint(
                      size: Size(900, totalHeight),
                      painter: BracketConnectionPainter(
                        round: 1,
                        matchPositions: round1Offsets,
                        nextRoundPositions: round2Offsets,
                        matchWidth: matchCardWidth,
                        matchHeight: 70, // Actual card height (without spacing)
                        horizontalSpacing: horizontalSpacing,
                        verticalOffset: 40, // Offset for the round labels
                      ),
                    ),

                  // Connection lines for Round 2 to Sweet 16
                  if (round2Matches.isNotEmpty && round3Matches.isNotEmpty)
                    CustomPaint(
                      size: Size(900, totalHeight),
                      painter: BracketConnectionPainter(
                        round: 2,
                        matchPositions: round2Offsets,
                        nextRoundPositions: round3Offsets,
                        matchWidth: matchCardWidth,
                        matchHeight: 70,
                        horizontalSpacing: horizontalSpacing,
                        verticalOffset: 40,
                      ),
                    ),

                  // Connection lines for Sweet 16 to Elite 8
                  if (round3Matches.isNotEmpty && round4Matches.isNotEmpty)
                    CustomPaint(
                      size: Size(900, totalHeight),
                      painter: BracketConnectionPainter(
                        round: 3,
                        matchPositions: round3Offsets,
                        nextRoundPositions: [round4Offset],
                        matchWidth: matchCardWidth,
                        matchHeight: 70,
                        horizontalSpacing: horizontalSpacing,
                        verticalOffset: 40,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildRegionHeader(String region) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      color: Colors.grey.shade100,
      child: Text(
        region,
        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
      ),
    );
  }

  Widget buildRoundLabel(String round, String dates) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text(
          round,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 14,
            color: Colors.black87,
          ),
          textAlign: TextAlign.center,
        ),
        Text(
          dates,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget buildMatchCard(Map<String, dynamic> match, bool isElite8) {
    // Convert scoreA and scoreB to String format for safe rendering
    final String scoreA = match['scoreA'].toString();
    final String scoreB = match['scoreB'].toString();

    // Determine if there's a winner (only if both scores are valid numbers)
    bool teamAWins = false;
    bool teamBWins = false;

    try {
      final int scoreANum = int.parse(scoreA);
      final int scoreBNum = int.parse(scoreB);
      teamAWins = scoreANum > scoreBNum;
      teamBWins = scoreBNum > scoreANum;
    } catch (e) {
      // Scores might be "--" or invalid numbers
    }

    return Container(
      width: 180,
      height: 83,
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(1),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Team A
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 7),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.grey.shade300)),
              color: teamAWins ? Colors.blue.shade50 : Colors.white,
            ),
            child: Row(
              children: [
                // Seed Box
                Container(
                  width: 26,
                  height: 26,
                  alignment: Alignment.center,
                  margin: const EdgeInsets.only(right: 8),
                  decoration: BoxDecoration(
                    color: _getTeamSeedColor(match['seedA'].toString()),
                    borderRadius: BorderRadius.circular(1),
                  ),
                  child: Text(
                    match['seedA'].toString(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                // Team Logo
                buildTeamLogo(match['teamA'].toString()),
                // Team Name
                Expanded(
                  child: Text(
                    match['teamA'].toString(),
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight:
                          teamAWins ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                ),
                // Score
                Container(
                  margin: const EdgeInsets.only(left: 4),
                  child: Text(
                    scoreA,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight:
                          teamAWins ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Team B
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 7),
            color: teamBWins ? Colors.blue.shade50 : Colors.white,
            child: Row(
              children: [
                // Seed Box
                Container(
                  width: 26,
                  height: 26,
                  alignment: Alignment.center,
                  margin: const EdgeInsets.only(right: 8),
                  decoration: BoxDecoration(
                    color: _getTeamSeedColor(match['seedB'].toString()),
                    borderRadius: BorderRadius.circular(1),
                  ),
                  child: Text(
                    match['seedB'].toString(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                // Team Logo
                buildTeamLogo(match['teamB'].toString()),
                // Team Name
                Expanded(
                  child: Text(
                    match['teamB'].toString(),
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight:
                          teamBWins ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                ),
                // Score
                Container(
                  margin: const EdgeInsets.only(left: 4),
                  child: Text(
                    scoreB,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight:
                          teamBWins ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // Team seed color helper
  Color _getTeamSeedColor(String seed) {
    // Convert seed to integer if possible
    int? seedNum;
    try {
      seedNum = int.parse(seed);
    } catch (e) {
      return Colors.grey; // Default color for non-numeric seeds
    }

    if (seedNum < 5) return Colors.blue.shade500;
    if (seedNum >= 5 && seedNum < 9) return Colors.green.shade800;
    if (seedNum >= 9 && seedNum < 13) return Colors.yellow.shade800;
    if (seedNum >= 13 && seedNum < 17) return Colors.red.shade800;

    return Colors.grey; // Default fallback
  }

  Widget buildBottomInfoBar() {
    return Container(
      color: Colors.grey.shade100,
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _buildInfoBarItem("1st Round", "Mar 21-22"),
            const SizedBox(width: 16),
            _buildInfoBarItem("2nd Round", "Mar 23-24"),
            const SizedBox(width: 16),
            _buildInfoBarItem("Sweet 16", "Mar 28-29"),
            const SizedBox(width: 16),
            _buildInfoBarItem("Elite 8", "Mar 30-31"),
            const SizedBox(width: 16),
            _buildInfoBarItem("Final Four", "Apr 6"),
            const SizedBox(width: 16),
            _buildInfoBarItem("Championship", "Apr 8"),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoBarItem(String round, String dates) {
    return Column(
      children: [
        Text(
          round,
          style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
        ),
        Text(
          dates,
          style: TextStyle(fontSize: 11, color: Colors.grey.shade700),
        ),
      ],
    );
  }
}

class BracketConnectionPainter extends CustomPainter {
  final int round;
  final List<double> matchPositions;
  final List<double> nextRoundPositions;
  final double matchWidth;
  final double matchHeight;
  final double horizontalSpacing;
  final double verticalOffset;

  BracketConnectionPainter({
    required this.round,
    required this.matchPositions,
    required this.nextRoundPositions,
    required this.matchWidth,
    required this.matchHeight,
    required this.horizontalSpacing,
    required this.verticalOffset,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint =
        Paint()
          ..color = Colors.grey.shade400
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.0;

    // Starting x position based on current round
    final double startX =
        round * (matchWidth + horizontalSpacing) - horizontalSpacing;
    final double endX = startX + horizontalSpacing;

    // For each pair of matches in current round
    for (int i = 0; i < matchPositions.length; i += 2) {
      if (i + 1 >= matchPositions.length) break; // Skip if we don't have a pair

      // Calculate the start and end y-positions with adjustment for card height
      final double y1 = verticalOffset + matchPositions[i] + (matchHeight / 2);
      final double y2 =
          verticalOffset + matchPositions[i + 1] + (matchHeight / 2);

      // Calculate the middle point between the two matches
      final double midY = (y1 + y2) / 2;

      // Find the corresponding next round match position
      final int nextRoundIndex = i ~/ 2;
      double nextMatchY =
          nextRoundIndex < nextRoundPositions.length
              ? verticalOffset +
                  nextRoundPositions[nextRoundIndex] +
                  (matchHeight / 2)
              : midY;

      // Lines from each match to the middle
      canvas.drawLine(Offset(startX, y1), Offset(startX + 10, y1), paint);
      canvas.drawLine(Offset(startX, y2), Offset(startX + 10, y2), paint);

      // Vertical connecting line
      canvas.drawLine(Offset(startX + 10, y1), Offset(startX + 10, y2), paint);

      // Line from the middle to the next round match
      canvas.drawLine(
        Offset(startX + 10, midY),
        Offset(endX, nextMatchY),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(BracketConnectionPainter oldDelegate) {
    return true; // Simple approach: always repaint
  }
}

Widget buildTeamLogo(String teamName, {bool small = false}) {
  // Normalize the team name to create a consistent filename format
  final normalizedName = _normalizeTeamName(teamName);
  final double size = small ? 18.0 : 24.0;

  return Container(
    width: size,
    height: size,
    margin: const EdgeInsets.only(right: 6),
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      color: Colors.white,
      border: Border.all(color: Colors.grey.shade300, width: 1),
    ),
    child: ClipOval(
      child: Image.asset(
        'assets/logos/$normalizedName.png',
        fit: BoxFit.contain,
        errorBuilder: (context, error, stackTrace) {
          // Fallback to a colored circle with first letter if image not found
          return Container(
            decoration: BoxDecoration(shape: BoxShape.circle),
            alignment: Alignment.center,
            child: Text(
              teamName.isNotEmpty ? teamName[0] : "",
              style: TextStyle(
                color: Colors.white,
                fontSize: small ? 8 : 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          );
        },
      ),
    ),
  );
}

// Helper function to normalize team names to match your filename format
String _normalizeTeamName(String teamName) {
  // Map of team name variations to their standardized filename
  final Map<String, String> teamNameMap = {
    'Abilene': 'abilene_christian_wildcats',
    'Air Force': 'air_force_falcons',
    'Akron': 'akron_zips',
    'Alabama': 'alabama_crimson_tide',
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
    'Army West Point': 'army_west_point_black_knights',
    'Atlantic 10': 'atlantic_10_conference',
    'Atlantic Coast': 'atlantic_coast_conference_acc',
    'Atlantic Sun': 'atlantic_sun_conference_asun',
    'Auburn': 'auburn_tigers',
    'Austin Peay': 'austin_peay_governors',
    'Ball State': 'ball_state_cardinals',
    'Baylor': 'baylor_bears',
    'Bellarmine': 'bellarmine_knights',
    'Belmont': 'belmont_bruins',
    'Bethune Cookman': 'bethune_cookman_wildcats',
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
    'UConn': 'connecticut_huskies',
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
    'Northern Illinois': 'northern_illinois_huskies',
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
    // Add more mappings as needed for your teams
  };

  // Look for exact matches first
  if (teamNameMap.containsKey(teamName)) {
    return teamNameMap[teamName]!;
  }

  // Try to find partial matches
  for (final entry in teamNameMap.entries) {
    if (teamName.contains(entry.key)) {
      return entry.value;
    }
  }

  // If no match found, create a normalized version of the name
  final normalized = teamName
      .toLowerCase()
      .replaceAll(RegExp(r'[^\w\s]'), '') // Remove special characters
      .replaceAll(RegExp(r'\s+'), '_'); // Replace spaces with underscores

  return normalized;
}
