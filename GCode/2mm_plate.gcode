; external perimeters extrusion width = 0.45mm
; perimeters extrusion width = 0.45mm
; infill extrusion width = 0.45mm
; solid infill extrusion width = 0.45mm
; top infill extrusion width = 0.40mm
; first layer extrusion width = 0.42mm

M73 P0 R18
M73 Q0 S18
M201 X1000 Y1000 Z1000 E5000 ; sets maximum accelerations, mm/sec^2
M203 X200 Y200 Z12 E120 ; sets maximum feedrates, mm/sec
M204 P1250 R1250 T1250 ; sets acceleration (P, T) and retract acceleration (R), mm/sec^2
M205 X8.00 Y8.00 Z0.40 E1.50 ; sets the jerk limits, mm/sec
M205 S0 T0 ; sets the minimum extruding and travel feed rate, mm/sec
M107
M115 U3.5.1 ; tell printer latest fw version
M83  ; extruder relative mode
M104 S220 ; set extruder temp
M140 S100 ; set bed temp
M190 S100 ; wait for bed temp
M109 S220 ; wait for extruder temp
G28 W ; home all without mesh bed level
G80 ; mesh bed leveling
G1 Y-3.0 F1000.0 ; go outside print area
G92 E0.0
G1 X60.0 E9.0  F1000.0 ; intro line
M73 Q0 S18
M73 P0 R18
G1 X100.0 E12.5  F1000.0 ; intro line
G92 E0.0
M221 S95
M900 K10; Filament gcode
G21 ; set units to millimeters
G90 ; use absolute coordinates
M83 ; use relative distances for extrusion
;BEFORE_LAYER_CHANGE
G92 E0.0
;0.2


G1 E-0.80000 F2100.00000
G1 Z0.600 F10800.000
;AFTER_LAYER_CHANGE
;0.2
G1 X99.127 Y77.993
G1 Z0.200
G1 E0.80000 F2100.00000
M204 S1000
G1 F1200; external perimeters extrusion width = 0.45mm
; perimeters extrusion width = 0.45mm
; infill extrusion width = 0.45mm
; solid infill extrusion width = 0.45mm
; top infill extrusion width = 0.40mm
; first layer extrusion width = 0.42mm

M73 P0 R18
M73 Q0 S18
M201 X1000 Y1000 Z1000 E5000 ; sets maximum accelerations, mm/sec^2
M203 X200 Y200 Z12 E120 ; sets maximum feedrates, mm/sec
M204 P1250 R1250 T1250 ; sets acceleration (P, T) and retract acceleration (R), mm/sec^2
M205 X8.00 Y8.00 Z0.40 E1.50 ; sets the jerk limits, mm/sec
M205 S0 T0 ; sets the minimum extruding and travel feed rate, mm/sec
M107
M115 U3.5.1 ; tell printer latest fw version
M83  ; extruder relative mode
M104 S220 ; set extruder temp
M140 S100 ; set bed temp
M190 S100 ; wait for bed temp
M109 S220 ; wait for extruder temp
G28 W ; home all without mesh bed level
G80 ; mesh bed leveling
G1 Y-3.0 F1000.0 ; go outside print area
G92 E0.0
G1 X60.0 E9.0  F1000.0 ; intro line
M73 Q0 S18
M73 P0 R18
G1 X100.0 E12.5  F1000.0 ; intro line
G92 E0.0
M221 S95
M900 K10; Filament gcode
G21 ; set units to millimeters
G90 ; use absolute coordinates
M83 ; use relative distances for extrusion
;BEFORE_LAYER_CHANGE
G92 E0.0
;0.2


G1 E-0.80000 F2100.00000
G1 Z0.600 F10800.000
;AFTER_LAYER_CHANGE
;0.2
G1 X99.127 Y77.993
G1 Z0.200
G1 E0.80000 F2100.00000
M204 S1000
G1 F1200abe; external perimeters extrusion width = 0.45mm
; perimeters extrusion width = 0.45mm
; infill extrusion width = 0.45mm
; solid infill extrusion width = 0.45mm
; top infill extrusion width = 0.40mm
; first layer extrusion width = 0.42mm

M73 P0 R18
M73 Q0 S18
M201 X1000 Y1000 Z1000 E5000 ; sets maximum accelerations, mm/sec^2
M203 X200 Y200 Z12 E120 ; sets maximum feedrates, mm/sec
M204 P1250 R1250 T1250 ; sets acceleration (P, T) and retract acceleration (R), mm/sec^2
M205 X8.00 Y8.00 Z0.40 E1.50 ; sets the jerk limits, mm/sec
M205 S0 T0 ; sets the minimum extruding and travel feed rate, mm/sec
M107
M115 U3.5.1 ; tell printer latest fw version
M83  ; extruder relative mode
M104 S220 ; set extruder temp
M140 S100 ; set bed temp
M190 S100 ; wait for bed temp
M109 S220 ; wait for extruder temp
G28 W ; home all without mesh bed level
G80 ; mesh bed leveling
G1 Y-3.0 F1000.0 ; go outside print area
G92 E0.0
G1 X60.0 E9.0  F1000.0 ; intro line
M73 Q0 S18
M73 P0 R18
G1 X100.0 E12.5  F1000.0 ; intro line
G92 E0.0
M221 S95
M900 K10; Filament gcode
G21 ; set units to millimeters
G90 ; use absolute coordinates
M83 ; use relative distances for extrusion
;BEFORE_LAYER_CHANGE
G92 E0.0
;0.2


G1 E-0.80000 F2100.00000
G1 Z0.600 F10800.000
;AFTER_LAYER_CHANGE
;0.2
G1 X99.127 Y77.993
G1 Z0.200
G1 E0.80000 F2100.00000
M204 S1000
G1 F1200
G1 X99.972 Y77.812 E0.02438
G1 X150.004 Y77.811 E1.41183
G1 X150.823 Y77.972 E0.02358
G1 X151.526 Y78.431 E0.02368
G1 X152.003 Y79.117 E0.02358
G1 X152.188 Y79.936 E0.02368
G1 X152.188 Y130.026 E1.41348
G1 X152.026 Y130.827 E0.02307
G1 X151.582 Y131.513 E0.02307
G1 X150.863 Y132.011 E0.02469
G1 X150.002 Y132.189 E0.02479
G1 X99.974 Y132.188 E1.41173
G1 X99.173 Y132.026 E0.02307
G1 X98.487 Y131.582 E0.02307
G1 X97.989 Y130.863 E0.02469
G1 X97.811 Y130.002 E0.02479
G1 X97.812 Y79.974 E1.41173
G1 X97.974 Y79.173 E0.02307
G1 X98.418 Y78.487 E0.02307
G1 X99.078 Y78.027 E0.02269
; PURGING FINISHED
G1 F8640
G1 X99.972 Y77.812 E-0.21237
G1 X102.344 Y77.812 E-0.54763
G1 E-0.04000 F2100.00000
G1 Z0.800 F10800.000
G1 X100.587 Y80.587
G1 Z0.200
G1 E0.80000 F2100.00000
G1 F1200
G1 X149.413 Y80.587 E1.37781
G1 X149.413 Y129.413 E1.37781
G1 X100.587 Y129.413 E1.37781
G1 X100.587 Y80.647 E1.37612
G1 X100.210 Y80.210 F10800.000
G1 F1200
G1 X149.790 Y80.210 E1.39909
G1 X149.790 Y129.790 E1.39909
G1 X100.210 Y129.790 E1.39909
G1 X100.210 Y80.270 E1.39740
G1 X100.596 Y80.314 F10800.000
G1 F8640
G1 X103.501 Y80.266 E-0.76000
G1 E-0.04000 F2100.00000
G1 Z0.800 F10800.000
G1 X101.464 Y129.300
G1 Z0.200
G1 E0.80000 F2100.00000
G1 F1200
G1 X100.870 Y128.705 E0.02378