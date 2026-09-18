# Music Platform Design

This is a fictional Spotify-like application design, not a Spotify annual report.

## Technology

The web client uses Next.js. The API services use Python. PostgreSQL stores
account and subscription records. Redis caches frequently requested data.

## Monetization

### Premium subscriptions

Premium listeners pay a monthly subscription. Premium includes ad-free playback,
higher audio quality, and offline downloads in the native application.

### Advertising

Free listeners hear audio advertisements. The advertising service manages audio
ad slots, targeting, and campaign analytics.

## Analytics

The dashboard tracks daily active users (DAU), monthly active users (MAU), and
subscription conversion. This design does not supply actual user counts.
