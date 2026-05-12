export class AppError extends Error {
  code: string;
  recoverable: boolean;
  setupRequired?: boolean;
  status: number;

  constructor(params: {
    code: string;
    message: string;
    recoverable?: boolean;
    setupRequired?: boolean;
    status?: number;
  }) {
    super(params.message);
    this.name = "AppError";
    this.code = params.code;
    this.recoverable = params.recoverable ?? true;
    this.setupRequired = params.setupRequired;
    this.status = params.status ?? 400;
  }
}

export function errorResponse(error: unknown) {
  if (error instanceof AppError) {
    return Response.json(
      {
        ok: false,
        error: {
          code: error.code,
          message: error.message,
          recoverable: error.recoverable,
          setupRequired: error.setupRequired
        }
      },
      { status: error.status }
    );
  }

  return Response.json(
    {
      ok: false,
      error: {
        code: "UNEXPECTED_ERROR",
        message:
          error instanceof Error
            ? `The request failed in a controlled error boundary: ${error.message}`
            : "The request failed in a controlled error boundary.",
        recoverable: true
      }
    },
    { status: 500 }
  );
}
