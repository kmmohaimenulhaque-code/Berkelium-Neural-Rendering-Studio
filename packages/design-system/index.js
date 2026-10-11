// Design tokens for Berkelium Studio
export const tokens = {
  // Color palette
  background: '#08090A',
  primaryText: '#F7F8F8',
  secondaryText: '#8A8F98',
  borders: '#1F2023',
  primaryAccent: '#00D4D4',
  
  // Spacing
  baseSpacingUnit: '8px',
  
  // Radius
  controlRadius: '6px',
  panelRadius: '8px'
};

export const theme = {
  colors: {
    background: tokens.background,
    primaryText: tokens.primaryText,
    secondaryText: tokens.secondaryText,
    borders: tokens.borders,
    primaryAccent: tokens.primaryAccent
  },
  spacing: {
    baseUnit: tokens.baseSpacingUnit
  },
  radius: {
    control: tokens.controlRadius,
    panel: tokens.panelRadius
  }
};